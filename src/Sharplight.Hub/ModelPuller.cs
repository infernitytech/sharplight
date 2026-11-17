using System.Diagnostics;
using Sharplight.Hub.Catalog;
using Sharplight.Package;
using Sharplight.Vision;

namespace Sharplight.Hub;

/// <summary>
/// Pulls a model from the Hub into a <see cref="ModelStore"/>: resolves the revision to a commit, chooses the variant's files,
/// checks the free disk space, downloads (resumable, verified), converts to a package with graph optimization, verifies that
/// the package opens, optionally precompiles it for a device, registers it (manifest written last) and deletes the sources.
/// An interrupted pull leaves the model unregistered; running it again resumes the downloads. Conversion goes through the
/// <see cref="ModelConverter"/>, so the formats and architectures it can convert are the ones registered with the container
/// (<c>services.AddSharplight()</c>).
/// </summary>
public sealed class ModelPuller
{
    private const long SpaceMargin = 256L << 20;
    private const double PackageSizeFactor = 1.05;

    /// <summary>The folder under <c>source/</c> that holds a catalog install's companions.</summary>
    private const string CompanionDirectoryName = "companions";

    private readonly HubClient _hub;
    private readonly FileDownloader _downloader;
    private readonly ModelStore _store;
    private readonly ModelConverter _converter;

    /// <summary>Creates a puller.</summary>
    /// <param name="hub">Resolves repositories and file addresses.</param>
    /// <param name="downloader">Downloads the files.</param>
    /// <param name="store">Receives the models.</param>
    /// <param name="converter">Converts the downloaded files to packages.</param>
    public ModelPuller(HubClient hub, FileDownloader downloader, ModelStore store, ModelConverter converter)
    {
        ArgumentNullException.ThrowIfNull(hub);
        ArgumentNullException.ThrowIfNull(downloader);
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(converter);
        _hub = hub;
        _downloader = downloader;
        _store = store;
        _converter = converter;
        Converter = ConvertAndVerify;
        CompositeConverter = ConvertWithCompanions;
    }

    /// <summary>Free bytes on the volume of a directory (replaceable for tests).</summary>
    internal Func<string, long> FreeSpace { get; init; } = directory => new DriveInfo(directory).AvailableFreeSpace;

    /// <summary>Builds and verifies the package: primary source, vision projector (or null), destination, log (replaceable for tests).</summary>
    internal Func<string, string?, string, Action<string>?, ConversionResult> Converter { get; init; }

    /// <summary>Builds and verifies a catalog install's package: primary source, companions by role, destination, log (replaceable for tests).</summary>
    internal Func<string, IReadOnlyList<KeyValuePair<string, string>>, string, Action<string>?, ConversionResult> CompositeConverter { get; init; }

    /// <summary>Pulls <paramref name="reference"/> unless it is already stored at the resolved commit.</summary>
    /// <param name="reference">The repository and optional variant tag.</param>
    /// <param name="options">Revision, precompile device, logging; or <see langword="null"/> for the defaults.</param>
    /// <param name="cancellationToken">Cancels downloads (they resume on the next pull) and the steps between them.</param>
    /// <returns>The stored model and timings.</returns>
    /// <exception cref="HubException">The repository is not accessible, has no matching variant, a download failed, the disk is too full, or the model cannot be converted.</exception>
    public async Task<PullResult> PullAsync(ModelReference reference, PullOptions? options = null, CancellationToken cancellationToken = default)
    {
        options ??= new PullOptions();
        var log = options.Log;
        log?.Invoke($"Resolving {reference} ({options.Revision})");
        var repository = await _hub.GetRepositoryAsync(reference.Repository, options.Revision, cancellationToken).ConfigureAwait(false);
        var selection = await SelectAsync(repository, reference.Tag, cancellationToken).ConfigureAwait(false);
        var name = selection.Tag.Length == 0 ? repository.Id : $"{repository.Id}:{selection.Tag}";
        var directory = _store.GetModelDirectory(repository.Id, selection.Tag);

        using var pullLock = ModelStore.Lock(directory, name);
        var existing = ModelStore.ReadManifest(directory);
        if (existing is not null && existing.Commit == repository.Commit && !options.Force && File.Exists(existing.PackagePath))
        {
            log?.Invoke($"{name} is up to date (commit {Short(repository.Commit)}).");
            return new PullResult(existing, true, 0, TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero);
        }

        if (repository.Gated && _hub.Options.Token is null)
        {
            throw new HubException($"{repository.Id} is a gated repository. Accept its terms on {new Uri(_hub.Options.Endpoint, repository.Id)}, "
                + $"create an access token and set {HubClientOptions.TokenVariable}.");
        }

        log?.Invoke($"Pulling {name} at commit {Short(repository.Commit)}: {selection.Files.Count} file(s), {ByteCount.Format(selection.TotalBytes)}");
        var sourceDirectory = PrepareDirectories(directory, repository.Commit);

        // Files go into a directory named like the repository: importers name checkpoint models after their directory.
        var filesDirectory = Path.Combine(sourceDirectory, RepoPath.Name(repository.Id));
        var planned = selection.Files.Select(file => new PlannedFile(repository, file, RepoPath.ToLocal(filesDirectory, file.Path))).ToList();
        EnsureFreeSpace(directory, planned);

        var stopwatch = Stopwatch.StartNew();
        long transferred = await DownloadAsync(planned, options, cancellationToken).ConfigureAwait(false);
        var downloadTime = stopwatch.Elapsed;

        cancellationToken.ThrowIfCancellationRequested();
        stopwatch.Restart();
        var pending = Path.Combine(directory, ModelStore.PendingPackageFileName);
        var primary = selection.PrimaryPath.Length == 0 ? filesDirectory : RepoPath.ToLocal(filesDirectory, selection.PrimaryPath);
        var projector = selection.ProjectorPath is { } projectorPath ? RepoPath.ToLocal(filesDirectory, projectorPath) : null;
        log?.Invoke($"Converting to {ModelStore.PackageFileName}");
        var conversion = await Task.Run(() => Converter(primary, projector, pending, log), cancellationToken).ConfigureAwait(false);
        var convertTime = stopwatch.Elapsed;
        log?.Invoke($"Built {conversion.ModelName} ({conversion.SourceArchitecture ?? conversion.Architecture}): {conversion.GraphCount} graph(s), {conversion.WeightCount} weights, "
            + $"{ByteCount.Format(conversion.PackageBytes)} [{convertTime.TotalSeconds:F1} s]");

        cancellationToken.ThrowIfCancellationRequested();
        stopwatch.Restart();
        var precompiled = options.Precompile is { } target ? await PrecompileAsync(pending, target, log, cancellationToken).ConfigureAwait(false) : [];
        var precompileTime = options.Precompile is null ? TimeSpan.Zero : stopwatch.Elapsed;

        var manifest = Register(directory, pending, repository, selection, conversion, precompiled, options);
        if (!options.KeepSource)
        {
            Directory.Delete(sourceDirectory, recursive: true);
        }

        log?.Invoke($"Stored {manifest.Name} as {manifest.ShortName} ({ByteCount.Format(manifest.PackageBytes)})");
        return new PullResult(manifest, false, transferred, downloadTime, convertTime, precompileTime);
    }

    /// <summary>
    /// Installs a verified-catalog variant: downloads exactly its pinned files and those of its companions (no Hub request is needed to
    /// choose them), converts them to one package with each companion in its auxiliary role, optionally precompiles it and registers it
    /// under <see cref="CatalogInstallPlan.Reference"/>. A model already installed from the same variant at the same commit is kept.
    /// </summary>
    /// <param name="plan">The variant and its companions (<see cref="CatalogInstallPlan.For"/>).</param>
    /// <param name="options">Precompile device, logging, progress (the revision is the catalog's); or <see langword="null"/>.</param>
    /// <param name="cancellationToken">Cancels downloads (they resume on the next install) and the steps between them.</param>
    /// <returns>The stored model and timings.</returns>
    /// <exception cref="HubException">A repository is gated without a token, a download failed or did not match its checksum, the disk is too full, or the model cannot be converted.</exception>
    public async Task<PullResult> InstallAsync(CatalogInstallPlan plan, PullOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(plan);
        options ??= new PullOptions();
        var log = options.Log;
        var variant = plan.Variant;
        var name = plan.Reference.ToString();
        var directory = _store.GetModelDirectory(variant.Repository, plan.StoreTag);

        using var pullLock = ModelStore.Lock(directory, name);
        var existing = ModelStore.ReadManifest(directory);
        if (existing is not null && plan.IsInstalledAs(existing) && !options.Force && File.Exists(existing.PackagePath))
        {
            log?.Invoke($"{plan.Entry.Name} ({variant.Label}) is installed (commit {Short(variant.Revision)}).");
            return new PullResult(existing, true, 0, TimeSpan.Zero, TimeSpan.Zero, TimeSpan.Zero);
        }

        if (plan.Entry.Gated && _hub.Options.Token is null)
        {
            throw new HubException($"{plan.Entry.Name} needs a gated repository. Accept its terms on {new Uri(_hub.Options.Endpoint, variant.Repository)}, "
                + $"create an access token and set {HubClientOptions.TokenVariable}.");
        }

        log?.Invoke($"Installing {plan.Entry.Name} ({variant.Label}) at commit {Short(variant.Revision)}: {plan.FileCount} file(s), {ByteCount.Format(plan.TotalBytes)}");
        var sourceDirectory = PrepareDirectories(directory, variant.Revision);
        var filesDirectory = Path.Combine(sourceDirectory, RepoPath.Name(variant.Repository));
        var planned = Plan(variant, filesDirectory).ToList();
        var auxiliary = new List<KeyValuePair<string, string>>();
        foreach (var companion in plan.Companions)
        {
            var companionDirectory = Path.Combine(sourceDirectory, CompanionDirectoryName, companion.Id, RepoPath.Name(companion.Repository));
            planned.AddRange(Plan(companion, companionDirectory));
            auxiliary.Add(new(companion.Role, PrimaryPath(companionDirectory, companion)));
        }

        EnsureFreeSpace(directory, planned);
        var stopwatch = Stopwatch.StartNew();
        long transferred = await DownloadAsync(planned, options, cancellationToken).ConfigureAwait(false);
        var downloadTime = stopwatch.Elapsed;

        cancellationToken.ThrowIfCancellationRequested();
        stopwatch.Restart();
        var pending = Path.Combine(directory, ModelStore.PendingPackageFileName);
        log?.Invoke($"Converting to {ModelStore.PackageFileName}" + (auxiliary.Count == 0 ? "" : $" with {string.Join(", ", auxiliary.Select(pair => pair.Key))}"));
        var conversion = await Task.Run(() => CompositeConverter(PrimaryPath(filesDirectory, variant), auxiliary, pending, log), cancellationToken).ConfigureAwait(false);
        var convertTime = stopwatch.Elapsed;

        cancellationToken.ThrowIfCancellationRequested();
        stopwatch.Restart();
        var precompiled = options.Precompile is { } target ? await PrecompileAsync(pending, target, log, cancellationToken).ConfigureAwait(false) : [];
        var precompileTime = options.Precompile is null ? TimeSpan.Zero : stopwatch.Elapsed;

        var sourceFiles = planned.Select(file => new ManifestFile
        {
            Path = Path.GetRelativePath(sourceDirectory, file.LocalPath).Replace(Path.DirectorySeparatorChar, '/'),
            Size = file.File.Size,
            Sha256 = file.File.Sha256,
        });
        var manifest = Register(directory, pending, packageBytes => new ModelManifest
        {
            PackageBytes = packageBytes,
            Repository = variant.Repository,
            Tag = plan.StoreTag,
            Alias = ModelManifest.AliasFor(variant.Repository),
            Revision = variant.Revision,
            Commit = variant.Revision,
            Format = CatalogFormats.ToModelFormat(variant.Format),
            ModelName = conversion.ModelName,
            Architecture = conversion.SourceArchitecture ?? plan.Entry.Architecture ?? conversion.Architecture,
            SourceFiles = [.. sourceFiles],
            SourceKept = options.KeepSource,
            Precompiled = precompiled,
            PulledAt = DateTimeOffset.UtcNow,
            Directory = directory,
            Catalog = plan.Key,
        });
        if (!options.KeepSource)
        {
            Directory.Delete(sourceDirectory, recursive: true);
        }

        log?.Invoke($"Stored {manifest.Name} as {manifest.ShortName} ({ByteCount.Format(manifest.PackageBytes)})");
        return new PullResult(manifest, false, transferred, downloadTime, convertTime, precompileTime);
    }

    /// <summary>The downloads of a catalog file set: its pinned files, checked against the catalog's sizes and checksums.</summary>
    private static IEnumerable<PlannedFile> Plan(CatalogFileSet set, string directory)
    {
        var repository = new HubRepository(set.Repository, set.Revision, set.Revision, false,
            [.. set.Files.Select(file => new HubFile(file.Path, file.Size, file.Sha256, file.GitOid))]);
        return repository.Files.Select(file => new PlannedFile(repository, file, RepoPath.ToLocal(directory, file.Path)));
    }

    private static string PrimaryPath(string directory, CatalogFileSet set) => set.Primary.Length == 0 ? directory : RepoPath.ToLocal(directory, set.Primary);

    private async Task<ModelSelection> SelectAsync(HubRepository repository, string? tag, CancellationToken cancellationToken)
    {
        var indexPath = ModelFileSelector.IndexPath(repository, tag);
        var index = indexPath is null ? null : await _hub.GetFileTextAsync(repository, indexPath, cancellationToken).ConfigureAwait(false);
        return ModelFileSelector.Select(repository, tag, index);
    }

    /// <summary>
    /// Creates the model and source directories. Sources of another commit (an interrupted pull of an older revision) are
    /// discarded, and so is a package left over from an interrupted conversion.
    /// </summary>
    private static string PrepareDirectories(string directory, string commit)
    {
        var sourceDirectory = Path.Combine(directory, ModelStore.SourceDirectoryName);
        var commitFile = Path.Combine(sourceDirectory, ModelStore.CommitFileName);
        if (Directory.Exists(sourceDirectory) && (!File.Exists(commitFile) || File.ReadAllText(commitFile).Trim() != commit))
        {
            Directory.Delete(sourceDirectory, recursive: true);
        }

        Directory.CreateDirectory(sourceDirectory);
        File.WriteAllText(commitFile, commit);
        File.Delete(Path.Combine(directory, ModelStore.PendingPackageFileName));
        return sourceDirectory;
    }

    /// <summary>Checks that the remaining downloads plus the package (written next to the sources) fit on the volume.</summary>
    private void EnsureFreeSpace(string directory, IReadOnlyList<PlannedFile> files)
    {
        long remaining = 0;
        foreach (var (_, file, local) in files)
        {
            long onDisk = File.Exists(local) ? new FileInfo(local).Length
                : File.Exists(local + DownloadRequest.PartialExtension) ? new FileInfo(local + DownloadRequest.PartialExtension).Length
                : 0;
            remaining += Math.Max(0, file.Size - onDisk);
        }

        long package = (long)(files.Sum(planned => planned.File.Size) * PackageSizeFactor);
        long needed = remaining + package + SpaceMargin;
        long free = FreeSpace(directory);
        if (free < needed)
        {
            throw new HubException($"Not enough disk space in {_store.Root}: {ByteCount.Format(needed)} needed ({ByteCount.Format(remaining)} to download, "
                + $"about {ByteCount.Format(package)} for the package), {ByteCount.Format(free)} free.");
        }
    }

    private async Task<long> DownloadAsync(IReadOnlyList<PlannedFile> files, PullOptions options, CancellationToken cancellationToken)
    {
        var progress = new AggregateProgress(options.Progress, files.Sum(planned => planned.File.Size));
        foreach (var (repository, file, local) in files)
        {
            var request = DownloadRequest.For(_hub, repository, file, local);
            await _downloader.DownloadAsync(request, progress, options.Log, cancellationToken).ConfigureAwait(false);
            progress.CompleteFile(file.Size);
        }

        return progress.Transferred;
    }

    private static async Task<IReadOnlyList<string>> PrecompileAsync(string package, PrecompileTarget target, Action<string>? log,
        CancellationToken cancellationToken)
    {
        try
        {
            var report = await Task.Run(() => PackagePrecompiler.PrecompileAsync(package, target.Providers, target.Options, target.Progress, cancellationToken),
                cancellationToken).ConfigureAwait(false);
            log?.Invoke($"Precompiled for {report.Devices.Count(d => d.Portable is not null || d.DeviceCode is not null)} device(s): "
                + $"{report.StoredSets.Count + report.UnchangedSets.Count} artifact set(s), {ByteCount.Format(report.StoredBytes)}");
            return [.. report.StoredSets, .. report.UnchangedSets];
        }
        catch (NotSupportedException error)
        {
            log?.Invoke($"Not precompiled: {error.Message}");
            return [];
        }
        catch (Exception error) when (error is not OperationCanceledException)
        {
            // A failed precompile (device unavailable, driver error) only costs compile time at the first load, unless it
            // damaged the package.
            log?.Invoke($"Precompiling failed ({error.Message}); kernels are compiled at the first load instead.");
            Verify(package);
            return [];
        }
    }

    private static ModelManifest Register(string directory, string pending, HubRepository repository, ModelSelection selection,
        ConversionResult conversion, IReadOnlyList<string> precompiled, PullOptions options)
    {
        return Register(directory, pending, packageBytes => new ModelManifest
        {
            PackageBytes = packageBytes,
            Repository = repository.Id,
            Tag = selection.Tag,
            Alias = ModelManifest.AliasFor(repository.Id),
            Revision = options.Revision,
            Commit = repository.Commit,
            Format = selection.Format,
            ModelName = conversion.ModelName,
            Architecture = repository.Architecture ?? conversion.SourceArchitecture ?? conversion.Architecture,
            SourceFiles = [.. selection.Files.Select(file => new ManifestFile { Path = file.Path, Size = file.Size, Sha256 = file.Sha256 })],
            SourceKept = options.KeepSource,
            Precompiled = precompiled,
            PulledAt = DateTimeOffset.UtcNow,
            Directory = directory,
        });
    }

    /// <summary>Moves the pending package into place and writes the manifest last (the model counts as stored only then).</summary>
    private static ModelManifest Register(string directory, string pending, Func<long, ModelManifest> manifest)
    {
        var packagePath = Path.Combine(directory, ModelStore.PackageFileName);
        File.Delete(Path.Combine(directory, ModelStore.ManifestFileName));
        File.Move(pending, packagePath, overwrite: true);
        var stored = manifest(new FileInfo(packagePath).Length);
        ModelStore.WriteManifest(stored);
        return stored;
    }

    private ConversionResult ConvertAndVerify(string primary, string? projector, string destination, Action<string>? log)
    {
        var options = new ConversionOptions { Log = log };
        ConversionResult result;
        try
        {
            result = projector is null
                ? _converter.Convert(primary, destination, options)
                : ConvertWithProjector(primary, projector, destination, options, log);
        }
        catch (NotSupportedException error)
        {
            throw new HubException($"Cannot convert the model: {error.Message}", inner: error);
        }

        Verify(destination);
        return result;
    }

    private ConversionResult ConvertWithCompanions(string primary, IReadOnlyList<KeyValuePair<string, string>> auxiliary, string destination, Action<string>? log)
    {
        ConversionResult result;
        try
        {
            result = _converter.Convert(primary, destination, new ConversionOptions { Log = log }, auxiliary.Count == 0 ? null : auxiliary);
        }
        catch (NotSupportedException error)
        {
            throw new HubException($"Cannot convert the model: {error.Message}", inner: error);
        }

        Verify(destination);
        return result;
    }

    private ConversionResult ConvertWithProjector(string primary, string projector, string destination, ConversionOptions options, Action<string>? log)
    {
        try
        {
            return _converter.Convert(primary, destination, options, [new(AuxiliaryRoles.Mmproj, projector)]);
        }
        catch (NotSupportedException error)
        {
            log?.Invoke($"The vision projector is not supported ({error.Message}); converting the language model alone.");
            return _converter.Convert(primary, destination, options);
        }
    }

    /// <summary>Checks that the package opens and holds at least one graph.</summary>
    private static void Verify(string package)
    {
        try
        {
            var info = ModelPackageReader.Open(package);
            if (info.Definition.Graphs.Count == 0)
            {
                throw new InvalidDataException("the package has no graphs");
            }
        }
        catch (InvalidDataException error)
        {
            throw new HubException($"The converted package does not open: {error.Message}", inner: error);
        }
    }

    private static string Short(string commit) => commit.Length > 12 ? commit[..12] : commit;
}
