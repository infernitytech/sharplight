using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Maintenance;

namespace Sharplight.Studio.ViewModels.Models;

// Export of a library model as a standalone .slm file, and import of .slm files or folders (the file picker or drag and drop onto the library).
internal sealed partial class LibraryViewModel
{
    [RelayCommand]
    private Task ExportAsync(LibraryEntryViewModel entry) => ExportAsync(entry, includePrecompiled: true, destination: null);

    [RelayCommand]
    private Task ExportLeanAsync(LibraryEntryViewModel entry) => ExportAsync(entry, includePrecompiled: false, destination: null);

    /// <summary>Writes the package of <paramref name="entry"/> to a standalone <c>.slm</c> file.</summary>
    /// <param name="entry">The model.</param>
    /// <param name="includePrecompiled">Keep the precompiled artifacts (larger file that loads fast); false drops them (they rebuild).</param>
    /// <param name="destination">The file to write, or null to ask with the save dialog.</param>
    public async Task ExportAsync(LibraryEntryViewModel entry, bool includePrecompiled, string? destination)
    {
        if (!entry.IsPackage)
        {
            Status = $"{entry.Name} is not a .slm package: convert it first.";
            return;
        }

        if (Operation.IsRunning)
        {
            return;
        }

        destination ??= await _services.Platform.PickSaveFileAsync("Export model", SafeFileName(entry.Id) + ".slm", s_packageFiles);
        if (destination is null)
        {
            return;
        }

        long size = 0;
        entry.IsBusy = true;
        try
        {
            bool completed = await Operation.RunAsync($"Exporting {entry.Name}", async (progress, cancellationToken) =>
                size = await _services.Maintenance.ExportAsync(entry.Entry, destination, includePrecompiled, progress, cancellationToken).ConfigureAwait(false));
            Status = completed
                ? $"Exported {entry.Name} to {destination} ({ByteFormat.Format(size)}{(includePrecompiled ? "" : ", without precompiled artifacts")})."
                : "Export cancelled; no file was written.";
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or InvalidOperationException or InvalidDataException)
        {
            Status = $"Export failed: {error.Message}";
        }
        finally
        {
            entry.IsBusy = false;
        }
    }

    [RelayCommand]
    private async Task ImportFileAsync()
    {
        var files = await _services.Platform.PickFilesAsync("Import .slm packages", [s_packageFiles], allowMultiple: true);
        await ImportPathsAsync(files);
    }

    [RelayCommand]
    private async Task ImportFolderAsync()
    {
        if (await _services.Platform.PickFolderAsync("Import a folder of .slm packages") is { } folder)
        {
            await ImportPathsAsync([folder]);
        }
    }

    /// <summary>
    /// Validates the <c>.slm</c> files at <paramref name="paths"/> (files, or folders searched for packages) with the package reader and asks
    /// before copying the valid ones into the model store; the others are reported with the reason.
    /// </summary>
    /// <param name="paths">Files and folders (from a picker or dropped onto the library).</param>
    public async Task ImportPathsAsync(IReadOnlyList<string> paths)
    {
        if (paths.Count == 0 || Operation.IsRunning)
        {
            return;
        }

        var candidates = new List<ImportCandidate>();
        bool completed = await Operation.RunAsync("Checking the packages", async (_, cancellationToken) =>
        {
            foreach (var path in paths)
            {
                candidates.AddRange(await _services.Maintenance.InspectForImportAsync(path, cancellationToken).ConfigureAwait(false));
            }
        });
        if (!completed)
        {
            Status = "Import cancelled.";
            return;
        }

        var valid = candidates.Where(candidate => candidate.IsValid).ToList();
        var rejected = candidates.Where(candidate => !candidate.IsValid).ToList();
        if (valid.Count == 0)
        {
            Status = "Nothing to import. " + RejectedText(rejected);
            return;
        }

        long bytes = valid.Sum(candidate => candidate.Bytes);
        var message = valid.Count == 1
            ? $"Import {valid[0].Name} ({ByteFormat.Format(bytes)}) into the model store? It was checked and is a valid package."
            : $"Import {valid.Count} packages ({ByteFormat.Format(bytes)}) into the model store? They were checked and are valid: {string.Join(", ", valid.Take(4).Select(candidate => candidate.Name))}{(valid.Count > 4 ? $" and {valid.Count - 4} more" : "")}.";
        if (rejected.Count > 0)
        {
            message += " " + RejectedText(rejected);
        }

        Ask(message, "Import", destructive: false, () => ImportConfirmedAsync(candidates));
    }

    private async Task ImportConfirmedAsync(IReadOnlyList<ImportCandidate> candidates)
    {
        ImportResult? result = null;
        try
        {
            bool completed = await Operation.RunAsync("Importing", async (progress, cancellationToken) =>
                result = await _services.Maintenance.ImportAsync(candidates, progress, cancellationToken).ConfigureAwait(false));
            if (!completed)
            {
                Status = "Import cancelled; nothing half copied was kept.";
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            Status = $"Import failed: {error.Message}";
        }

        await RefreshAsync();
        if (result is { } done)
        {
            Status = $"Imported {ByteFormat.Count(done.Imported.Count, "model", "models")}." + (done.Rejected.Count > 0 ? " " + RejectedText(done.Rejected) : "");
        }
    }

    private static string RejectedText(IReadOnlyList<ImportCandidate> rejected)
        => rejected.Count == 0 ? "" : "Not imported: " + string.Join("; ", rejected.Take(3).Select(candidate => $"{Path.GetFileName(candidate.Path)}: {candidate.Error}")) + (rejected.Count > 3 ? $" and {rejected.Count - 3} more." : "");

    private static string SafeFileName(string name) => string.Concat(name.Select(c => Path.GetInvalidFileNameChars().Contains(c) || c == ':' ? '-' : c));
}
