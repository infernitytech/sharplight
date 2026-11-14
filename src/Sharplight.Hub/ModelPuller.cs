using System.Diagnostics;
using Sharplight.Package;
using Sharplight.Package.Import;
using Sharplight.Vision;

namespace Sharplight.Hub;

/// <summary>
/// Pulls a model from the Hub into a <see cref="ModelStore"/>: resolves the revision to a commit, chooses the variant's files,
/// checks the free disk space, downloads (resumable, verified), converts to a package with graph optimization, verifies that
/// the package opens, optionally precompiles it for a device, registers it (manifest written last) and deletes the sources.
/// An interrupted pull leaves the model unregistered; running it again resumes the downloads. The host registers the
/// importers and adapters for the formats it wants to convert (e.g. <c>GgufImporter.Register()</c>).
/// </summary>
public sealed class ModelPuller
{
    private const long SpaceMargin = 256L << 20;
    private const double PackageSizeFactor = 1.05;

    private readonly HubClient _hub;
    private readonly FileDownloader _downloader;
    private readonly ModelStore _store;

    /// <summary>Creates a puller.</summary>
    /// <param name="hub">Resolves repositories and file addresses.</param>
    /// <param name="downloader">Downloads the files.</param>
    /// <param name="store">Receives the models.</param>
    public ModelPuller(HubClient hub, FileDownloader downloader, ModelStore store)
    {
        _hub = hub;
        _downloader = downloader;
        _store = store;
    }

    /// <summary>Free bytes on the volume of a directory (replaceable for tests).</summary>
    internal Func<string, long> FreeSpace { get; init; } = directory => new DriveInfo(directory).AvailableFreeSpace;

    /// <summary>Builds and verifies the package: primary source, vision projector (or null), destination, log (replaceable for tests).</summary>
    internal Func<string, string?, string, Action<string>?, ConversionResult> Converter { get; init; } = ConvertAndVerify;

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
        EnsureFreeSpace(directory, filesDirectory, selection);

        var stopwatch = Stopwatch.StartNew();
        long transferred = await DownloadAsync(repository, selection, filesDirectory, options, cancellationToken).ConfigureAwait(false);
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
        var precompiled = options.Precompile is { } target ? await PrecompileAsync(pending, target, log).ConfigureAwait(false) : [];
        var precompileTime = options.Precompile is null ? TimeSpan.Zero : stopwatch.Elapsed;

        var manifest = Register(directory, pending, repository, selection, conversion, precompiled, options);
        if (!options.KeepSource)
        {
            Directory.Delete(sourceDirectory, recursive: true);
        }

        log?.Invoke($"Stored {manifest.Name} as {manifest.ShortName} ({ByteCount.Format(manifest.PackageBytes)})");
        return new PullResult(manifest, false, transferred, downloadTime, convertTime, precompileTime);
    }

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
    private void EnsureFreeSpace(string directory, string sourceDirectory, ModelSelection selection)
    {
        long remaining = 0;
        foreach (var file in selection.Files)
        {
            var local = RepoPath.ToLocal(sourceDirectory, file.Path);
            long onDisk = File.Exists(local) ? new FileInfo(local).Length
                : File.Exists(local + DownloadRequest.PartialExtension) ? new FileInfo(local + DownloadRequest.PartialExtension).Length
                : 0;
            remaining += Math.Max(0, file.Size - onDisk);
        }

        long package = (long)(selection.TotalBytes * PackageSizeFactor);
        long needed = remaining + package + SpaceMargin;
        long free = FreeSpace(directory);
        if (free < needed)
        {
            throw new HubException($"Not enough disk space in {_store.Root}: {ByteCount.Format(needed)} needed ({ByteCount.Format(remaining)} to download, "
                + $"about {ByteCount.Format(package)} for the package), {ByteCount.Format(free)} free.");
        }
    }

    private async Task<long> DownloadAsync(HubRepository repository, ModelSelection selection, string sourceDirectory, PullOptions options,
        CancellationToken cancellationToken)
    {
        var progress = new AggregateProgress(options.Progress, selection.TotalBytes);
        foreach (var file in selection.Files)
        {
            var request = DownloadRequest.For(_hub, repository, file, RepoPath.ToLocal(sourceDirectory, file.Path));
            await _downloader.DownloadAsync(request, progress, options.Log, cancellationToken).ConfigureAwait(false);
            progress.CompleteFile(file.Size);
        }

        return progress.Transferred;
    }

    private static async Task<IReadOnlyList<string>> PrecompileAsync(string package, PrecompileTarget target, Action<string>? log)
    {
        try
        {
            var result = await Task.Run(() => PackagePrecompiler.Precompile(package, target.Provider, target.Device, target.Options)).ConfigureAwait(false);
            log?.Invoke($"Precompiled for {result.Device}: {result.StoredSets.Count + result.UnchangedSets.Count} artifact set(s), {ByteCount.Format(result.StoredBytes)}");
            return [.. result.StoredSets, .. result.UnchangedSets];
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
        var packagePath = Path.Combine(directory, ModelStore.PackageFileName);
        File.Delete(Path.Combine(directory, ModelStore.ManifestFileName));
        File.Move(pending, packagePath, overwrite: true);
        var manifest = new ModelManifest
        {
            Repository = repository.Id,
            Tag = selection.Tag,
            Alias = ModelManifest.AliasFor(repository.Id),
            Revision = options.Revision,
            Commit = repository.Commit,
            Format = selection.Format,
            ModelName = conversion.ModelName,
            Architecture = repository.Architecture ?? conversion.SourceArchitecture ?? conversion.Architecture,
            PackageBytes = new FileInfo(packagePath).Length,
            SourceFiles = [.. selection.Files.Select(file => new ManifestFile { Path = file.Path, Size = file.Size, Sha256 = file.Sha256 })],
            SourceKept = options.KeepSource,
            Precompiled = precompiled,
            PulledAt = DateTimeOffset.UtcNow,
            Directory = directory,
        };
        ModelStore.WriteManifest(manifest);
        return manifest;
    }

    private static ConversionResult ConvertAndVerify(string primary, string? projector, string destination, Action<string>? log)
    {
        var options = new ConversionOptions { Log = log };
        ConversionResult result;
        try
        {
            result = projector is null
                ? ModelConverter.Convert(primary, destination, options)
                : ConvertWithProjector(primary, projector, destination, options, log);
        }
        catch (NotSupportedException error)
        {
            throw new HubException($"Cannot convert the model: {error.Message}", inner: error);
        }

        Verify(destination);
        return result;
    }

    private static ConversionResult ConvertWithProjector(string primary, string projector, string destination, ConversionOptions options, Action<string>? log)
    {
        try
        {
            return ModelConverter.Convert(primary, destination, options, [(AuxiliaryRoles.Mmproj, projector)]);
        }
        catch (NotSupportedException error)
        {
            log?.Invoke($"The vision projector is not supported ({error.Message}); converting the language model alone.");
            return ModelConverter.Convert(primary, destination, options);
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
