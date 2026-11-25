namespace Sharplight.Studio.Services.Maintenance;

/// <summary>A package found for import and what the inspector made of it.</summary>
/// <param name="Path">The <c>.slm</c> file.</param>
/// <param name="Name">The model name.</param>
/// <param name="Bytes">File size.</param>
/// <param name="Error">Why it cannot be imported (not a package, unreadable, truncated), or null when it is valid.</param>
internal sealed record ImportCandidate(string Path, string Name, long Bytes, string? Error)
{
    /// <summary>The package passed validation.</summary>
    public bool IsValid => Error is null;
}
