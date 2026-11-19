using Sharplight.Studio.Services.Library;

namespace Sharplight.Studio.Services.Maintenance;

/// <summary>
/// Disk usage, cleanup, export and import of library models. Everything runs off the calling thread, reports progress and honours
/// cancellation; nothing outside the storage locations in effect is ever touched, and a resident model is never deleted or rewritten.
/// </summary>
internal interface IModelMaintenance
{
    /// <summary>Measures every storage location and finds the partial downloads.</summary>
    public Task<StorageOverview> GetOverviewAsync(IProgress<MaintenanceProgress>? progress, CancellationToken cancellationToken);

    /// <summary>Where the bytes of a model are (reads the package's section table, not its weights).</summary>
    public Task<ModelSizeBreakdown> GetBreakdownAsync(LibraryEntry entry, CancellationToken cancellationToken);

    /// <summary>Bytes of a cache, or of the given caches together.</summary>
    public Task<long> MeasureCachesAsync(CacheKinds kinds, CancellationToken cancellationToken);

    /// <summary>Whether <paramref name="entry"/> is resident or being precompiled, so it must not be deleted or rewritten; the reason, or null.</summary>
    public string? WhyInUse(LibraryEntry entry);

    /// <summary>Bytes a deletion of <paramref name="entry"/> frees (nothing for a file outside the store, which is only removed from the list).</summary>
    public long FreedByDelete(LibraryEntry entry);

    /// <summary>Bytes removing the precompiled artifacts of <paramref name="entry"/> frees.</summary>
    public Task<long> FreedByStripAsync(LibraryEntry entry, CancellationToken cancellationToken);

    /// <summary>Deletes a stored model (an added file is only removed from the list).</summary>
    /// <returns>The bytes freed.</returns>
    /// <exception cref="ModelInUseException">The model is resident or being precompiled.</exception>
    public Task<long> DeleteAsync(LibraryEntry entry, CancellationToken cancellationToken);

    /// <summary>Removes the precompiled artifacts from a package and compacts it; they are rebuilt by precompiling or at the next load.</summary>
    /// <returns>The bytes freed.</returns>
    /// <exception cref="ModelInUseException">The model is resident or being precompiled.</exception>
    public Task<long> StripPrecompiledAsync(LibraryEntry entry, CancellationToken cancellationToken);

    /// <summary>Deletes the files of the given caches.</summary>
    /// <returns>The bytes freed.</returns>
    public Task<long> ClearCachesAsync(CacheKinds kinds, IProgress<MaintenanceProgress>? progress, CancellationToken cancellationToken);

    /// <summary>Deletes partial downloads (the caller makes sure no download is running).</summary>
    /// <returns>The bytes freed.</returns>
    public Task<long> RemovePartialDownloadsAsync(IReadOnlyList<PartialDownload> items, CancellationToken cancellationToken);

    /// <summary>Copies a library package to <paramref name="destination"/> as a standalone <c>.slm</c> file.</summary>
    /// <param name="entry">The model (a <c>.slm</c> package).</param>
    /// <param name="destination">The file to write (replaced atomically).</param>
    /// <param name="includePrecompiled">Keep the precompiled artifacts; false writes the package without them (smaller; they rebuild).</param>
    /// <param name="progress">Receives the bytes copied.</param>
    /// <param name="cancellationToken">Cancels; no partial file is left.</param>
    /// <returns>The size of the exported file.</returns>
    public Task<long> ExportAsync(LibraryEntry entry, string destination, bool includePrecompiled, IProgress<MaintenanceProgress>? progress, CancellationToken cancellationToken);

    /// <summary>Finds the <c>.slm</c> files at <paramref name="path"/> (a file, or a folder searched two levels deep) and validates each with the package reader.</summary>
    public Task<IReadOnlyList<ImportCandidate>> InspectForImportAsync(string path, CancellationToken cancellationToken);

    /// <summary>Copies the valid candidates into the model store and registers them; invalid ones are reported and skipped.</summary>
    public Task<ImportResult> ImportAsync(IReadOnlyList<ImportCandidate> candidates, IProgress<MaintenanceProgress>? progress, CancellationToken cancellationToken);
}
