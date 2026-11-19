namespace Sharplight.Studio.Services.Maintenance;

/// <summary>The outcome of importing files or folders.</summary>
/// <param name="Imported">The packages now in the library (paths in the model store).</param>
/// <param name="Rejected">The files that failed validation, with the reason.</param>
internal sealed record ImportResult(IReadOnlyList<string> Imported, IReadOnlyList<ImportCandidate> Rejected);
