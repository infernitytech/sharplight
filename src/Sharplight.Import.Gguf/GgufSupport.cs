using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.Gguf;

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
    /// Files of this architecture are speculative-decoding drafters (EAGLE-3, Gemma 4 assistants, e.g. the <c>MTP/</c> files of a GGUF repository):
    /// they are loaded as the draft source of the model they were trained for, not as models of their own.
    /// </summary>
    /// <param name="ggufArchitecture">The <c>general.architecture</c> value.</param>
    public static bool IsDraftModel(string? ggufArchitecture) => GgufArchitectureMappings.Find(ggufArchitecture)?.IsDraftModel == true;

    /// <summary>
    /// The element type of a ggml tensor type id, or null when the id is unknown to this version (or was removed from ggml); files
    /// with such tensors do not import. A known type may still lack compute kernels (e.g. <c>Q8_K</c>, an activation-only type).
    /// </summary>
    /// <param name="ggmlType">The type id as the tensor table stores it.</param>
    public static DataType? GetTensorType(int ggmlType) => GgmlTypes.TryGetDataType(ggmlType);

    /// <summary>
    /// Checks a GGUF file against its importer mapping from its header alone (e.g. read before the file is downloaded): the mapping's
    /// feature checks (<c>Validate</c>), its hyper-parameters, and the canonical name of every tensor, the same steps the importer runs.
    /// Tensor data is never read; a step that would need it leaves the check incomplete.
    /// </summary>
    /// <param name="metadata">The metadata of the (first) file, as <see cref="GgufHeaderProbe.ReadMetadata"/> returns it.</param>
    /// <param name="tensors">The tensor table of every shard of the model (see <see cref="GgufHeaderProbe.ReadTensorTable"/>).</param>
    public static GgufImportCheck Validate(IReadOnlyDictionary<string, object> metadata, IReadOnlyList<GgufTensorEntry> tensors)
    {
        ArgumentNullException.ThrowIfNull(metadata);
        ArgumentNullException.ThrowIfNull(tensors);
        if (metadata.GetValueOrDefault("general.architecture") is not string architecture)
        {
            return new GgufImportCheck(["The GGUF metadata names no general.architecture."], IsComplete: true);
        }

        if (GgufArchitectureMappings.Find(architecture) is not { } mapping)
        {
            return new GgufImportCheck([$"GGUF architecture '{architecture}' has no canonical mapping."], IsComplete: true);
        }

        // A split model declares the tensor count of all its shards: fewer entries mean a shard is missing (an incomplete upload).
        if (metadata.GetValueOrDefault("split.tensors.count") is IConvertible declared && Convert.ToInt64(declared, System.Globalization.CultureInfo.InvariantCulture) is var total
            && tensors.Count < total)
        {
            string shards = metadata.GetValueOrDefault("split.count") is IConvertible count ? $"{Convert.ToInt64(count, System.Globalization.CultureInfo.InvariantCulture)} files" : "several files";
            return new GgufImportCheck([$"The model is split into {shards} holding {total} tensors, but the files present hold {tensors.Count}: a shard is missing."],
                IsComplete: true);
        }

        var infos = new List<GgufTensorInfo>(tensors.Count);
        var unknownTypes = new SortedSet<int>();
        foreach (var tensor in tensors)
        {
            if (GgmlTypes.TryGetDataType(tensor.GgmlType) is { } type)
            {
                infos.Add(GgufTensorInfo.FromHeader(tensor, type));
            }
            else
            {
                unknownTypes.Add(tensor.GgmlType);
            }
        }

        if (unknownTypes.Count > 0)
        {
            return new GgufImportCheck([$"Tensors use ggml type ids unknown to this version: {string.Join(", ", unknownTypes)}."], IsComplete: true);
        }

        var byName = new Dictionary<string, GgufTensorInfo>(StringComparer.Ordinal);
        foreach (var info in infos)
        {
            byName.TryAdd(info.Name, info);
        }

        var view = new GgufMetadataView(metadata, architecture, name => byName.GetValueOrDefault(name));
        bool nextnOnly = GgufTensorNames.HoldsOnlyNextnBlocks(byName.Keys, view);   // an mtp-*.gguf draft file, whatever its mapping imports
        try
        {
            if (mapping.Validate(view).ToList() is { Count: > 0 } problems)
            {
                return new GgufImportCheck(problems, IsComplete: true, IsDraftModel: mapping.IsDraftModel);
            }

            var (_, _, weights, _) = GgufImporter.Map("", metadata, infos, mapping, view);
            if (IsDraftLayersOnly(weights.Keys))
            {
                return new GgufImportCheck(
                    ["The file holds only multi-token-prediction layers (an MTP draft file): it is loaded as the draft source of the model it was made for."],
                    IsComplete: true, IsDraftModel: true);
            }

            if (mapping.IsDraftModel)
            {
                return new GgufImportCheck(
                    [$"The file is a speculative-decoding drafter (GGUF architecture '{architecture}'): it maps, and is loaded as the draft source of the model it was made for."],
                    IsComplete: true, IsDraftModel: true);
            }

            return new GgufImportCheck([], IsComplete: true);
        }
        catch (GgufHeaderOnlyDataException)
        {
            return new GgufImportCheck([], IsComplete: false);
        }
        catch (Exception error) when (error is not OutOfMemoryException)
        {
            // Every other failure of these steps fails the import the same way.
            return new GgufImportCheck([error.Message.TrimStart(':', ' ')], IsComplete: true, IsDraftModel: nextnOnly || mapping.IsDraftModel);
        }
    }

    /// <summary>Canonical weights with multi-token-prediction layers (<c>mtp.layers.*</c>) but no trunk layers (<c>layers.*</c>).</summary>
    private static bool IsDraftLayersOnly(IEnumerable<string> canonicalNames)
    {
        bool draft = false;
        foreach (string name in canonicalNames)
        {
            if (name.StartsWith("layers.", StringComparison.Ordinal))
            {
                return false;
            }

            draft |= name.StartsWith("mtp.layers.", StringComparison.Ordinal);
        }

        return draft;
    }
}
