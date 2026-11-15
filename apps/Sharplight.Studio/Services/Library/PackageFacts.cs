using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Library;

/// <summary>What the library shows about a model file without loading it (see <see cref="ModelInspector"/>).</summary>
/// <param name="Capabilities">What it can do (from the package's graphs and chat template).</param>
/// <param name="ContextLength">Trained context length, when known.</param>
/// <param name="Architecture">Architecture, when known.</param>
/// <param name="Precompiled">Descriptors (formatted) of the precompiled artifact sets stored in the package.</param>
internal sealed record PackageFacts(ModelCapabilities Capabilities, int? ContextLength, string? Architecture, IReadOnlyList<string> Precompiled)
{
    /// <summary>Nothing known (not a package, or unreadable).</summary>
    public static PackageFacts None { get; } = new(ModelCapabilities.None, null, null, []);
}
