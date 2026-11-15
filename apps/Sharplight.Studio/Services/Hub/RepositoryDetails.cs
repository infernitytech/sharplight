using Sharplight.Hub;

namespace Sharplight.Studio.Services.Hub;

/// <summary>A repository's files and pullable variants.</summary>
/// <param name="Repository">Commit, files and metadata.</param>
/// <param name="Format">The format it is pulled as, or null when it has no supported model files.</param>
/// <param name="Variants">Variants, recommended first.</param>
/// <param name="Layout">
/// The multi-component layout the repository has when it cannot be pulled yet (<see cref="ModelFormat.Diffusers"/>,
/// <see cref="ModelFormat.WhisperCpp"/>, <see cref="ModelFormat.Slm"/>), or null.
/// </param>
/// <param name="Components">The parts of a multi-component repository (a Diffusers pipeline's folders, whisper.cpp's model files), shown as one model.</param>
internal sealed record RepositoryDetails(HubRepository Repository, ModelFormat? Format, IReadOnlyList<VariantInfo> Variants,
    ModelFormat? Layout = null, IReadOnlyList<HubComponent>? Components = null)
{
    /// <summary>The parts of a multi-component repository (never null).</summary>
    public IReadOnlyList<HubComponent> Parts => Components ?? [];
}
