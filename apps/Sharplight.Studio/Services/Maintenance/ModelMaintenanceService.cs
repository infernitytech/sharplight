using Sharplight.Hub;
using Sharplight.Package;
using Sharplight.Runtime;
using Sharplight.Storage;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Precompile;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Settings;

namespace Sharplight.Studio.Services.Maintenance;

/// <summary>
/// <see cref="IModelMaintenance"/> over the model store, the storage layout in effect and the runtime (which says what is resident).
/// Deletion only happens inside the model store and the cache folders of the layout; links are never followed.
/// </summary>
internal sealed class ModelMaintenanceService : IModelMaintenance
{
    private const int CopyBuffer = 1 << 20;
    private static readonly string[] s_partialSuffixes = [".partial", ".pending", ".part", ".tmp"];
    private static readonly string[] s_partialNames = [".pull.lock", ".commit"];

    private readonly ISettingsService _settings;
    private readonly IStorageLocationService _storage;
    private readonly IModelRuntime _runtime;
    private readonly IModelLibrary _library;
    private readonly IModelUsageStore _usage;
    private readonly IPrecompileService _precompiler;

    public ModelMaintenanceService(ISettingsService settings, IStorageLocationService storage, IModelRuntime runtime, IModelLibrary library,
        IModelUsageStore usage, IPrecompileService precompiler)
    {
        _settings = settings;
        _storage = storage;
        _runtime = runtime;
        _library = library;
        _usage = usage;
        _precompiler = precompiler;
    }

    /// <inheritdoc/>
    public Task<StorageOverview> GetOverviewAsync(IProgress<MaintenanceProgress>? progress, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            var layout = _storage.Current;
            var store = _settings.StoreRoot;
            var data = _storage.Paths.DataDirectory;
            var cache = layout.CacheDirectory;
            var logs = layout.LogsDirectory;
            var locations = new List<StorageLocationUsage>();

            progress?.Report(new MaintenanceProgress("Measuring the model store", 0, 4));
            var partials = FindPartials(store, cancellationToken);
            long storeTotal = FileTree.Size(store, null, cancellationToken);
            long packages = 0;
            long artifacts = 0;
            foreach (var manifest in new ModelStore(store).List())
            {
                cancellationToken.ThrowIfCancellationRequested();
                var (size, set) = PackageParts(manifest.PackagePath);
                packages += size;
                artifacts += set;
            }

            long partial = partials.Sum(item => item.Bytes);
            long models = Math.Max(0, packages - artifacts);
            long other = Math.Max(0, storeTotal - models - artifacts - partial);
            locations.Add(Location("Model store", store, [
                new StorageUsageItem(UsageKeys.Models, "Models", models),
                new StorageUsageItem(UsageKeys.PackageArtifacts, "Precompiled in packages", artifacts),
                new StorageUsageItem(UsageKeys.Downloads, "Downloads in progress", partial),
                new StorageUsageItem(UsageKeys.Other, "Other", other),
            ]));

            progress?.Report(new MaintenanceProgress("Measuring the data folder", 1, 4));
            var inData = new[] { store, cache, logs }.Where(folder => FileTree.IsInside(folder, data)).ToList();
            long outputs = FileTree.Size(_storage.Paths.PagesDirectory, inData, cancellationToken) + FileTree.Size(_storage.Paths.GenerationsDirectory, inData, cancellationToken);
            long dataTotal = FileTree.Size(data, inData, cancellationToken);
            locations.Add(Location("Studio data", data, [
                new StorageUsageItem(UsageKeys.Outputs, "Outputs and run histories", Math.Min(outputs, dataTotal)),
                new StorageUsageItem(UsageKeys.Other, "Conversations, settings, feed cache", Math.Max(0, dataTotal - outputs)),
            ]));

            progress?.Report(new MaintenanceProgress("Measuring the caches", 2, 4));
            long kernels = FileTree.Size(layout.ArtifactsDirectory, null, cancellationToken);
            long shaders = FileTree.Size(layout.ShaderSourcesDirectory, null, cancellationToken);
            long cacheTotal = FileTree.Size(cache, null, cancellationToken);
            locations.Add(Location("Caches", cache, [
                new StorageUsageItem(UsageKeys.KernelCache, "Precompiled kernels (PTX, cubin, SPIR-V device stages)", kernels),
                new StorageUsageItem(UsageKeys.ShaderCache, "Shader sources", shaders),
                new StorageUsageItem(UsageKeys.MediaCache, "Media and other caches", Math.Max(0, cacheTotal - kernels - shaders)),
            ]));

            progress?.Report(new MaintenanceProgress("Measuring the logs", 3, 4));
            locations.Add(Location("Logs", logs, [new StorageUsageItem(UsageKeys.Logs, "Logs", FileTree.Size(logs, null, cancellationToken))]));
            progress?.Report(new MaintenanceProgress("Done", 4, 4));
            return new StorageOverview(locations, partials);
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public Task<ModelSizeBreakdown> GetBreakdownAsync(LibraryEntry entry, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            long total = SizeOf(entry);
            if (!entry.Path.EndsWith(ModelPackage.Extension, StringComparison.OrdinalIgnoreCase) || !File.Exists(entry.Path))
            {
                return new ModelSizeBreakdown(total, 0, 0, [], 0, IsPackage: false);
            }

            try
            {
                var info = ModelPackageReader.Open(entry.Path);
                var artifacts = info.ArtifactSets
                    .GroupBy(set => (set.Provider, Device: DeviceOf(set)))
                    .Select(group => new ArtifactUsage(group.Key.Provider, group.Key.Device,
                        string.Join(", ", group.Select(set => set.Kind).Where(kind => kind.Length > 0).Distinct()), group.Sum(set => set.Length), group.Sum(set => set.EntryCount)))
                    .OrderByDescending(item => item.Bytes)
                    .ToList();
                return new ModelSizeBreakdown(info.FileSize, info.WeightBytes, info.Attachments.Sum(item => item.Length), artifacts, info.ReclaimableBytes, IsPackage: true);
            }
            catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException)
            {
                return new ModelSizeBreakdown(total, 0, 0, [], 0, IsPackage: false);
            }
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public Task<long> MeasureCachesAsync(CacheKinds kinds, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            var layout = _storage.Current;
            long total = 0;
            foreach (var (folder, keep) in CacheFolders(layout, kinds))
            {
                total += FileTree.Size(folder, keep, cancellationToken);
            }

            return total;
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public string? WhyInUse(LibraryEntry entry)
    {
        if (_runtime.LoadedModels.Any(model => string.Equals(model.Path, entry.Path, StringComparison.Ordinal)))
        {
            return $"{entry.DisplayName} is resident. Unload it first.";
        }

        if (_precompiler.IsBusy(entry.Path))
        {
            return $"{entry.DisplayName} is being precompiled. Cancel that first.";
        }

        return null;
    }

    /// <inheritdoc/>
    public long FreedByDelete(LibraryEntry entry) => entry.Manifest is { } manifest ? FileTree.Size(manifest.Directory) : 0;

    /// <inheritdoc/>
    public async Task<long> FreedByStripAsync(LibraryEntry entry, CancellationToken cancellationToken)
    {
        var breakdown = await GetBreakdownAsync(entry, cancellationToken).ConfigureAwait(false);
        return breakdown.ArtifactBytes + breakdown.ReclaimableBytes;
    }

    /// <inheritdoc/>
    public async Task<long> DeleteAsync(LibraryEntry entry, CancellationToken cancellationToken)
    {
        if (WhyInUse(entry) is { } reason)
        {
            throw new ModelInUseException(reason);
        }

        long freed = await Task.Run(() => FreedByDelete(entry), cancellationToken).ConfigureAwait(false);
        await _library.RemoveAsync(entry).ConfigureAwait(false);
        _usage.Forget(entry.Path);
        return freed;
    }

    /// <inheritdoc/>
    public Task<long> StripPrecompiledAsync(LibraryEntry entry, CancellationToken cancellationToken)
    {
        if (WhyInUse(entry) is { } reason)
        {
            throw new ModelInUseException(reason);
        }

        return Task.Run(() =>
        {
            cancellationToken.ThrowIfCancellationRequested();
            long before = new FileInfo(entry.Path).Length;
            ModelPackage.Strip(entry.Path);
            ModelPackage.Compact(entry.Path);
            return Math.Max(0, before - new FileInfo(entry.Path).Length);
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public Task<long> ClearCachesAsync(CacheKinds kinds, IProgress<MaintenanceProgress>? progress, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            long freed = 0;
            var folders = CacheFolders(_storage.Current, kinds);
            for (int i = 0; i < folders.Count; i++)
            {
                progress?.Report(new MaintenanceProgress($"Clearing {Path.GetFileName(folders[i].Folder)}", i, folders.Count));
                freed += FileTree.Empty(folders[i].Folder, folders[i].Keep, cancellationToken);
            }

            progress?.Report(new MaintenanceProgress("Done", folders.Count, folders.Count));
            return freed;
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public Task<long> RemovePartialDownloadsAsync(IReadOnlyList<PartialDownload> items, CancellationToken cancellationToken)
    {
        return Task.Run(() =>
        {
            var store = new ModelStore(_settings.StoreRoot);
            long freed = 0;
            foreach (var item in items)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!FileTree.IsInside(item.Path, store.Root) || string.Equals(Path.GetFullPath(item.Path), store.Root, StringComparison.Ordinal))
                {
                    continue;
                }

                try
                {
                    if (item.IsDirectory && Directory.Exists(item.Path))
                    {
                        if (HasManifestBelow(item.Path))
                        {
                            continue;
                        }

                        freed += FileTree.Size(item.Path);
                        Directory.Delete(item.Path, recursive: true);
                        PruneEmptyParents(store.Root, Path.GetDirectoryName(item.Path)!);
                    }
                    else if (File.Exists(item.Path))
                    {
                        long length = new FileInfo(item.Path).Length;
                        File.Delete(item.Path);
                        freed += length;
                    }
                }
                catch (Exception error) when (error is IOException or UnauthorizedAccessException)
                {
                    // A download that is still writing keeps its file; it is not partial after all.
                }
            }

            return freed;
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public async Task<long> ExportAsync(LibraryEntry entry, string destination, bool includePrecompiled, IProgress<MaintenanceProgress>? progress, CancellationToken cancellationToken)
    {
        if (!entry.Path.EndsWith(ModelPackage.Extension, StringComparison.OrdinalIgnoreCase) || !File.Exists(entry.Path))
        {
            throw new InvalidOperationException($"{entry.DisplayName} is not a .slm package: convert it first.");
        }

        var target = Path.GetFullPath(destination);
        if (string.Equals(target, Path.GetFullPath(entry.Path), StringComparison.Ordinal))
        {
            throw new InvalidOperationException("Choose another file than the model itself.");
        }

        var temporary = target + ".export";
        try
        {
            await CopyAsync(entry.Path, temporary, $"Exporting {entry.DisplayName}", progress, cancellationToken).ConfigureAwait(false);
            await Task.Run(() =>
            {
                if (!includePrecompiled)
                {
                    progress?.Report(new MaintenanceProgress("Removing the precompiled artifacts", 0, 0));
                    ModelPackage.Strip(temporary);
                    ModelPackage.Compact(temporary);
                }

                ModelPackageReader.Open(temporary);
                File.Move(temporary, target, overwrite: true);
            }, cancellationToken).ConfigureAwait(false);
            return new FileInfo(target).Length;
        }
        finally
        {
            TryDelete(temporary);
        }
    }

    /// <inheritdoc/>
    public Task<IReadOnlyList<ImportCandidate>> InspectForImportAsync(string path, CancellationToken cancellationToken)
    {
        return Task.Run<IReadOnlyList<ImportCandidate>>(() =>
        {
            IEnumerable<string> files;
            if (Directory.Exists(path))
            {
                files = Directory.EnumerateFiles(path, "*" + ModelPackage.Extension, new EnumerationOptions { RecurseSubdirectories = true, MaxRecursionDepth = 2, IgnoreInaccessible = true, AttributesToSkip = FileAttributes.ReparsePoint })
                    .Order(StringComparer.Ordinal).Take(500);
            }
            else
            {
                files = [path];
            }

            var found = new List<ImportCandidate>();
            foreach (var file in files)
            {
                cancellationToken.ThrowIfCancellationRequested();
                found.Add(Validate(file));
            }

            if (found.Count == 0)
            {
                found.Add(new ImportCandidate(path, Path.GetFileName(path), 0, Directory.Exists(path) ? "The folder has no .slm package." : "The file does not exist."));
            }

            return found;
        }, cancellationToken);
    }

    /// <inheritdoc/>
    public async Task<ImportResult> ImportAsync(IReadOnlyList<ImportCandidate> candidates, IProgress<MaintenanceProgress>? progress, CancellationToken cancellationToken)
    {
        var imported = new List<string>();
        var rejected = candidates.Where(candidate => !candidate.IsValid).ToList();
        var store = new ModelStore(_settings.StoreRoot);
        var valid = candidates.Where(candidate => candidate.IsValid).ToList();
        for (int i = 0; i < valid.Count; i++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var candidate = valid[i];
            if (FileTree.IsInside(candidate.Path, store.Root))
            {
                rejected.Add(candidate with { Error = "It is already in the model store." });
                continue;
            }

            var step = new Progress<MaintenanceProgress>(update => progress?.Report(update with { Message = $"{update.Message} ({i + 1} of {valid.Count})" }));
            var slug = Slug(Path.GetFileNameWithoutExtension(candidate.Path));
            var repository = "imported/" + slug;
            for (int suffix = 2; store.List().Any(model => model.Repository == repository); suffix++)
            {
                repository = $"imported/{slug}-{suffix}";
            }

            var directory = store.GetModelDirectory(repository, "");
            Directory.CreateDirectory(directory);
            var target = Path.Combine(directory, ModelStore.PackageFileName);
            var pending = target + ".pending";
            try
            {
                await CopyAsync(candidate.Path, pending, $"Importing {candidate.Name}", step, cancellationToken).ConfigureAwait(false);
                var info = await Task.Run(() =>
                {
                    var read = ModelPackageReader.Open(pending);
                    _ = read.Definition;
                    File.Move(pending, target, overwrite: true);
                    return read;
                }, cancellationToken).ConfigureAwait(false);
                store.Save(new ModelManifest
                {
                    Repository = repository,
                    Alias = repository["imported/".Length..],
                    Commit = "imported",
                    Format = ModelFormat.Slm,
                    ModelName = info.Definition.Name,
                    Architecture = info.Definition.Architecture,
                    PackageBytes = info.FileSize,
                    PulledAt = DateTimeOffset.UtcNow,
                    Directory = directory,
                });
                imported.Add(target);
            }
            catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException)
            {
                rejected.Add(candidate with { Error = error.Message });
                TryDelete(pending);
                TryDeleteEmptyDirectory(directory);
            }
            catch (OperationCanceledException)
            {
                TryDelete(pending);
                TryDeleteEmptyDirectory(directory);
                throw;
            }
        }

        return new ImportResult(imported, rejected);
    }

    private static ImportCandidate Validate(string file)
    {
        var name = Path.GetFileNameWithoutExtension(file);
        long bytes = 0;
        try
        {
            if (!File.Exists(file))
            {
                return new ImportCandidate(file, name, 0, "The file does not exist.");
            }

            bytes = new FileInfo(file).Length;
            if (!file.EndsWith(ModelPackage.Extension, StringComparison.OrdinalIgnoreCase))
            {
                return new ImportCandidate(file, name, bytes, "Only .slm packages can be imported; convert other models first.");
            }

            var info = ModelPackageReader.Open(file);
            var definition = info.Definition;
            return new ImportCandidate(file, string.IsNullOrWhiteSpace(definition.Name) ? name : definition.Name, bytes, null);
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException or NotSupportedException)
        {
            return new ImportCandidate(file, name, bytes, $"Not a valid package: {error.Message}");
        }
    }

    private static async Task CopyAsync(string source, string destination, string message, IProgress<MaintenanceProgress>? progress, CancellationToken cancellationToken)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(destination))!);
        await using var input = new FileStream(source, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete, CopyBuffer, FileOptions.Asynchronous | FileOptions.SequentialScan);
        await using var output = new FileStream(destination, FileMode.Create, FileAccess.Write, FileShare.None, CopyBuffer, FileOptions.Asynchronous);
        var buffer = new byte[CopyBuffer];
        long done = 0;
        long total = input.Length;
        int read;
        while ((read = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false)) > 0)
        {
            await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
            done += read;
            progress?.Report(new MaintenanceProgress(message, done, total));
        }
    }

    private static string Slug(string name)
    {
        var chars = name.ToLowerInvariant().Select(c => char.IsAsciiLetterOrDigit(c) || c is '.' or '_' or '-' ? c : '-').ToArray();
        var slug = new string(chars).Trim('-', '.');
        return slug.Length == 0 ? "model" : slug;
    }

    private static void TryDelete(string path)
    {
        try
        {
            File.Delete(path);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // A leftover temporary file is listed as a partial download.
        }
    }

    private static void TryDeleteEmptyDirectory(string directory)
    {
        try
        {
            if (Directory.Exists(directory) && !Directory.EnumerateFileSystemEntries(directory).Any())
            {
                Directory.Delete(directory);
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // Left in place.
        }
    }

    private static StorageLocationUsage Location(string title, string path, IReadOnlyList<StorageUsageItem> items)
    {
        long total = 0;
        long free = 0;
        try
        {
            var probe = Path.GetFullPath(path);
            while (!Directory.Exists(probe) && Path.GetDirectoryName(probe) is { } parent)
            {
                probe = parent;
            }

            var drive = new DriveInfo(probe);
            total = drive.TotalSize;
            free = drive.AvailableFreeSpace;
        }
        catch (Exception error) when (error is IOException or ArgumentException or UnauthorizedAccessException)
        {
            // Unknown volume figures are shown as such.
        }

        return new StorageLocationUsage(title, path, total, free, items);
    }

    private static List<(string Folder, IReadOnlyCollection<string>? Keep)> CacheFolders(StorageLayout layout, CacheKinds kinds)
    {
        var folders = new List<(string, IReadOnlyCollection<string>?)>();
        if (kinds.HasFlag(CacheKinds.Kernels))
        {
            folders.Add((layout.ArtifactsDirectory, null));
        }

        if (kinds.HasFlag(CacheKinds.Shaders))
        {
            folders.Add((layout.ShaderSourcesDirectory, null));
        }

        if (kinds.HasFlag(CacheKinds.Media))
        {
            folders.Add((layout.CacheDirectory, [layout.ArtifactsDirectory, layout.ShaderSourcesDirectory]));
        }

        return folders;
    }

    private static long SizeOf(LibraryEntry entry)
    {
        try
        {
            return File.Exists(entry.Path) ? new FileInfo(entry.Path).Length : FileTree.Size(entry.Path);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return entry.SizeBytes;
        }
    }

    private static (long Size, long Artifacts) PackageParts(string path)
    {
        try
        {
            var info = ModelPackageReader.Open(path);
            return (info.FileSize, info.ArtifactSets.Sum(set => set.Length));
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            return (File.Exists(path) ? new FileInfo(path).Length : 0, 0);
        }
    }

    private static string DeviceOf(ArtifactSetInfo set)
    {
        foreach (var key in new[] { ArtifactDescriptorKeys.DeviceName, ArtifactDescriptorKeys.Device, ArtifactDescriptorKeys.Target, ArtifactDescriptorKeys.Vendor })
        {
            if (set.Descriptor.TryGetValue(key, out var value) && value.Length > 0)
            {
                return value;
            }
        }

        return "";
    }

    private static bool HasManifestBelow(string directory)
    {
        return Directory.EnumerateFiles(directory, ModelStore.ManifestFileName, new EnumerationOptions { RecurseSubdirectories = true, IgnoreInaccessible = true, AttributesToSkip = FileAttributes.ReparsePoint }).Any();
    }

    /// <summary>Partial downloads: model folders without a manifest (interrupted pulls) and partial or pending files next to a model.</summary>
    private static List<PartialDownload> FindPartials(string root, CancellationToken cancellationToken)
    {
        var found = new List<PartialDownload>();
        if (!Directory.Exists(root))
        {
            return found;
        }

        var store = new ModelStore(root);
        var registered = store.List().Select(model => Path.GetFullPath(model.Directory)).ToHashSet(StringComparer.Ordinal);
        foreach (var model in registered)
        {
            foreach (var file in Directory.EnumerateFiles(model))
            {
                var name = Path.GetFileName(file);
                if (s_partialSuffixes.Any(suffix => name.EndsWith(suffix, StringComparison.OrdinalIgnoreCase)) || s_partialNames.Contains(name))
                {
                    found.Add(new PartialDownload(file, new FileInfo(file).Length, IsDirectory: false));
                }
            }
        }

        foreach (var directory in DirectoriesAt(root, 3))
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!registered.Contains(Path.GetFullPath(directory)) && Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories).Any())
            {
                found.Add(new PartialDownload(directory, FileTree.Size(directory, null, cancellationToken), IsDirectory: true));
            }
        }

        return found;
    }

    private static IEnumerable<string> DirectoriesAt(string root, int depth)
    {
        IEnumerable<string> level = [root];
        for (int i = 0; i < depth; i++)
        {
            level = level.SelectMany(directory =>
            {
                try
                {
                    return Directory.EnumerateDirectories(directory).Where(child => (File.GetAttributes(child) & FileAttributes.ReparsePoint) == 0).ToList();
                }
                catch (Exception error) when (error is IOException or UnauthorizedAccessException)
                {
                    return [];
                }
            });
        }

        return level;
    }

    private static void PruneEmptyParents(string root, string directory)
    {
        var full = Path.GetFullPath(root).TrimEnd(Path.DirectorySeparatorChar);
        for (var current = Path.GetFullPath(directory); current.Length > full.Length && FileTree.IsInside(current, full); current = Path.GetDirectoryName(current)!)
        {
            if (Directory.EnumerateFileSystemEntries(current).Any())
            {
                return;
            }

            Directory.Delete(current);
        }
    }
}
