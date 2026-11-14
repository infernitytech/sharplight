using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// What the GGUF importer accepts, asked before a file is opened (e.g. from the metadata <see cref="GgufHeaderProbe"/> reads out of the
/// start of a file that is not downloaded yet). The answers come from the registered architecture mappings, so they follow every
/// mapping that is added.
/// </summary>
public static class GgufSupport
{
    /// <summary>Every <c>general.architecture</c> value the importer maps, in no particular order.</summary>
    public static IReadOnlyCollection<string> Architectures => [.. GgufArchitectureMappings.All.SelectMany(mapping => mapping.GgufArchitectures)];

    /// <summary>How far an architecture is supported: unsupported without a mapping, partial when its mapping names a restriction.</summary>
    /// <param name="ggufArchitecture">The <c>general.architecture</c> value.</param>
    public static SupportLevel GetArchitectureSupport(string? ggufArchitecture)
    {
        if (GgufArchitectureMappings.Find(ggufArchitecture) is not { } mapping)
        {
            return SupportLevel.Unsupported;
        }

        return mapping.PartialSupport(ggufArchitecture!) is null ? SupportLevel.Supported : SupportLevel.Partial;
    }

    /// <summary>What restricts a partially supported architecture, or null when it is fully supported or not supported at all.</summary>
    /// <param name="ggufArchitecture">The <c>general.architecture</c> value.</param>
    public static string? GetPartialSupportNote(string? ggufArchitecture)
        => GgufArchitectureMappings.Find(ggufArchitecture)?.PartialSupport(ggufArchitecture!);

    /// <summary>
    /// Files of this architecture are multimodal projectors (llama.cpp's <c>clip</c> mmproj files): they are loaded next to the language
    /// model they belong to, not as models of their own.
    /// </summary>
    /// <param name="ggufArchitecture">The <c>general.architecture</c> value.</param>
    public static bool IsProjector(string? ggufArchitecture) => GgufArchitectureMappings.Find(ggufArchitecture)?.IsProjector == true;

    /// <summary>
    /// The element type of a ggml tensor type id, or null when the id is unknown to this version (or was removed from ggml); files
    /// with such tensors do not import. A known type may still lack compute kernels (e.g. <c>Q8_K</c>, an activation-only type).
    /// </summary>
    /// <param name="ggmlType">The type id as the tensor table stores it.</param>
    public static DataType? GetTensorType(int ggmlType) => GgmlTypes.TryGetDataType(ggmlType);
}
