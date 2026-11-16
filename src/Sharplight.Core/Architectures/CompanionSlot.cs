namespace Sharplight.Architectures;

/// <summary>
/// Where a companion file of a transformer-only diffusion checkpoint goes (<see cref="DiffusionCompanionAdapter"/>): the architectures
/// it accepts, the canonical weight prefix it replaces and the renaming of its weights, the metadata it brings and the family keys the
/// primary keeps.
/// </summary>
/// <param name="Architectures">The companion source architectures accepted.</param>
/// <param name="WeightPrefix">The canonical weight prefix the companion replaces.</param>
/// <param name="Rename">The canonical name of a companion weight, or null to drop it.</param>
/// <param name="SourceMetadataPrefix">The prefix of the companion's metadata keys that are copied.</param>
/// <param name="MetadataKey">The component key the metadata goes under (<c>{arch}.{key}.</c>).</param>
/// <param name="TokenizerMetadataPrefix">Where the companion's tokenizer goes ("" = the canonical keys), or null to drop it.</param>
/// <param name="FamilyKeys">Keys under the component that the primary source keeps.</param>
public sealed record CompanionSlot(IReadOnlyList<string> Architectures, string WeightPrefix, Func<string, string?> Rename, string SourceMetadataPrefix,
    string MetadataKey, string? TokenizerMetadataPrefix, IReadOnlyList<string> FamilyKeys)
{
    /// <summary>Whether a companion of <paramref name="architecture"/> fills this slot.</summary>
    /// <param name="architecture">The companion source's architecture.</param>
    public bool Accepts(string architecture) => Architectures.Contains(architecture);
}
