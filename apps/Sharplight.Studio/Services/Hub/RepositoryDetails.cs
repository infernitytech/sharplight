using Sharplight.Hub;

namespace Sharplight.Studio.Services.Hub;

/// <summary>A repository's files and pullable variants.</summary>
/// <param name="Repository">Commit, files and metadata.</param>
/// <param name="Format">The format it is pulled as, or null when it has no supported model files.</param>
/// <param name="Variants">Variants, recommended first.</param>
internal sealed record RepositoryDetails(HubRepository Repository, ModelFormat? Format, IReadOnlyList<VariantInfo> Variants);
