using System.Globalization;
using System.Text.Json.Nodes;
using Sharplight.Cpu;
using Sharplight.Gguf;
using Sharplight.Hub;
using Sharplight.HuggingFace;
using Sharplight.Models;
using Sharplight.Onnx;
using Sharplight.Tensors;

namespace Sharplight.Studio.Services.Compatibility;

/// <summary>
/// Judges whether a model is likely to import and run from the metadata that can be read before downloading it (a GGUF header and
/// tensor table, a checkpoint's <c>config.json</c>, an ONNX model's opset imports, the Hub's search summary). The answers come from
/// the importers (<see cref="GgufSupport"/>, <see cref="HfSupport"/>) and the CPU kernels every model can fall back to, so nothing is
/// listed here.
/// </summary>
internal static class CompatibilityChecker
{
    /// <summary>Tensor types named in a reason before the rest are summarized.</summary>
    private const int ListedTypes = 6;

    /// <summary>Problems named in a reason before the rest are summarized.</summary>
    private const int ListedProblemCharacters = 600;

    /// <summary>
    /// The verdict for GGUF metadata and, when the whole tensor table was read (of every shard), its tensors: their types must be known
    /// and have kernels, and the importer's mapping must accept the file (<see cref="GgufSupport.Validate"/>: its hyper-parameters and
    /// features and the name of every tensor). Without the tensors the verdict is on the architecture alone.
    /// </summary>
    /// <param name="metadata">Key/values as <see cref="GgufHeaderProbe.ReadMetadata"/> returns them.</param>
    /// <param name="tensors">The tensor table of every shard (see <see cref="GgufHeaderProbe.ReadTensorTable"/>), or null when not read.</param>
    public static ModelCompatibility FromGguf(IReadOnlyDictionary<string, object> metadata, IReadOnlyList<GgufTensorEntry>? tensors)
    {
        if (!metadata.TryGetValue("general.architecture", out var value) || value is not string architecture)
        {
            return ModelCompatibility.Unknown("The GGUF header could not be read, so the architecture is not known.");
        }

        if (GgufSupport.IsProjector(architecture))
        {
            return new ModelCompatibility(CompatibilityState.Projector, architecture,
                $"A multimodal projector (mmproj, GGUF architecture '{architecture}'), not a model: it is loaded together with the language model it belongs to.");
        }

        if (GgufSupport.IsDraftModel(architecture))
        {
            return DraftModel(architecture, $"GGUF architecture '{architecture}'");
        }

        var level = GgufSupport.GetArchitectureSupport(architecture);
        if (level == SupportLevel.Unsupported)
        {
            return new ModelCompatibility(CompatibilityState.UnsupportedArchitecture, architecture,
                $"No importer mapping exists for the GGUF architecture '{architecture}', so importing it fails.");
        }

        var tensorTypes = tensors is null ? null : GgufHeaderProbe.CountTypes(tensors);
        if (tensorTypes is not null && UnsupportedTensorType(tensorTypes) is { } unsupported)
        {
            return new ModelCompatibility(CompatibilityState.UnsupportedWeightFormat, unsupported.Name,
                $"{Count(unsupported.Tensors, "tensor")} stored as {unsupported.Name}, which has no compute kernels, so loading fails. "
                + $"Tensor types: {DescribeTypes(tensorTypes)}.");
        }

        if (tensors is not null)
        {
            var check = GgufSupport.Validate(metadata, tensors);
            if (check.IsDraftModel)
            {
                return DraftModel(architecture, $"multi-token-prediction layers of GGUF architecture '{architecture}'");
            }

            if (check.Problems.Count > 0)
            {
                return new ModelCompatibility(CompatibilityState.UnsupportedVariant, architecture,
                    $"The GGUF architecture '{architecture}' is mapped, but importing this file fails: {Shorten(string.Join(" ", check.Problems))}");
            }

            if (!check.IsComplete)
            {
                return ModelCompatibility.Unknown(
                    $"The GGUF architecture '{architecture}' is mapped, but part of the import needs the weights themselves, so it is checked when the model is imported.");
            }
        }

        string types = tensorTypes is null
            ? "The tensors are read when this quantization is selected."
            : $"All {tensors!.Count} tensors map to the importer, with kernels for every type: {DescribeTypes(tensorTypes)}.";
        return level == SupportLevel.Partial
            ? new ModelCompatibility(CompatibilityState.Partial, architecture,
                $"The GGUF architecture '{architecture}' is partially supported: {GgufSupport.GetPartialSupportNote(architecture)}. {types}", tensors is not null)
            : new ModelCompatibility(CompatibilityState.Supported, architecture,
                $"The GGUF architecture '{architecture}' is supported. {types}", tensors is not null);
    }

    /// <summary>The verdict for a Hugging Face checkpoint's configuration.</summary>
    /// <param name="configJson">The text of <c>config.json</c>.</param>
    /// <param name="quantizationConfigJson">The text of <c>hf_quant_config.json</c> (ModelOpt), or null.</param>
    /// <param name="validate">
    /// Also run the importer mapping's checks of the hyper-parameters and features (<see cref="HfSupport.Validate"/>); false for a
    /// configuration that holds only the model type and quantization (from the search summary).
    /// </param>
    public static ModelCompatibility FromConfig(string configJson, string? quantizationConfigJson = null, bool validate = true)
    {
        if (HfSupport.GetModelType(configJson) is not { } modelType)
        {
            return ModelCompatibility.Unknown("The checkpoint's config.json names no model_type, so the architecture is not known.");
        }

        if (ArchitectureOnly(modelType) is { IsUnsupported: true } unsupported)
        {
            return unsupported;
        }

        if (HfSupport.IsDraftModel(modelType))
        {
            return DraftModel(modelType, $"model type '{modelType}'");
        }

        var quantization = HfSupport.GetQuantizationSupport(configJson, quantizationConfigJson);
        var verdict = WithQuantization(modelType, quantization.Method, quantization.Level);
        // Only a verdict that says the checkpoint imports is checked against the importer (an unknown one stays unknown).
        if (!validate || verdict.State is not (CompatibilityState.Supported or CompatibilityState.Partial)
            || HfSupport.Validate(configJson) is not { Count: > 0 } problems)
        {
            return verdict;
        }

        return new ModelCompatibility(CompatibilityState.UnsupportedVariant, modelType,
            $"The model type '{modelType}' is mapped, but importing this checkpoint fails: {Shorten(string.Join(" ", problems))}");
    }

    /// <summary>The verdict for an ONNX model: its opset imports are reported; operators are checked at import.</summary>
    /// <param name="opsets">The opset imports, or null when they could not be read.</param>
    public static ModelCompatibility FromOnnx(IReadOnlyList<OnnxOpset>? opsets)
    {
        string imports = opsets is { Count: > 0 }
            ? "Opsets: " + string.Join(", ", opsets.Select(opset => string.Create(CultureInfo.InvariantCulture, $"{opset.Domain} {opset.Version}"))) + ". "
            : "The opset imports could not be read before downloading. ";
        return new ModelCompatibility(CompatibilityState.CheckedAtImport, null,
            imports + "ONNX models are not mapped per architecture: each operator is checked when the model is imported.", opsets is not null);
    }

    /// <summary>
    /// The verdict for a search result from what the Hub's search reports (the GGUF architecture, or a checkpoint's model type and
    /// quantization method). Weight formats of GGUF files are checked per quantization in the repository details.
    /// </summary>
    /// <param name="summary">The search result.</param>
    public static ModelCompatibility FromSummary(HubModelSummary summary)
    {
        if (summary.HasProjectorOnly)
        {
            return new ModelCompatibility(CompatibilityState.Projector, summary.GgufArchitecture,
                "The repository holds only multimodal projectors (mmproj), not models: each is loaded together with the language model it belongs to.");
        }

        if (summary.Formats.Contains(ModelFormat.Gguf))
        {
            return summary.GgufArchitecture is { } architecture
                ? FromGguf(new Dictionary<string, object> { ["general.architecture"] = architecture }, null) with { WeightsChecked = false }
                : ModelCompatibility.Unknown("The Hub reports no GGUF architecture for this repository; it is read when the repository is opened.");
        }

        if (summary.Formats.Contains(ModelFormat.Safetensors))
        {
            if (summary.ModelType is not { } modelType)
            {
                return ModelCompatibility.Unknown("The Hub reports no model_type for this repository; config.json is read when the repository is opened.");
            }

            var config = new JsonObject { ["model_type"] = modelType };
            if (summary.QuantizationMethod is { } method)
            {
                config["quantization_config"] = method == "mlx" ? new JsonObject { ["bits"] = 4 } : new JsonObject { ["quant_method"] = method };
            }

            var verdict = FromConfig(config.ToJsonString(), validate: false);
            return verdict.IsUnsupported ? verdict : verdict with { WeightsChecked = false };
        }

        return summary.Formats.Contains(ModelFormat.Onnx)
            ? FromOnnx(null) with { WeightsChecked = false }
            : ModelCompatibility.Unknown("The repository has no GGUF, safetensors or ONNX model files.");
    }

    /// <summary>The verdict for a speculative-decoding drafter: it runs, but only next to the model it was trained for.</summary>
    private static ModelCompatibility DraftModel(string architecture, string what)
        => new(CompatibilityState.DraftModel, architecture,
            $"A speculative-decoding draft model ({what}), not a model of its own: it is loaded together with the model it drafts for.");

    /// <summary>The verdict for the architecture of a checkpoint's model type alone.</summary>
    private static ModelCompatibility ArchitectureOnly(string modelType)
    {
        return HfSupport.GetArchitectureSupport(modelType) switch
        {
            SupportLevel.Unsupported => new ModelCompatibility(CompatibilityState.UnsupportedArchitecture, modelType,
                $"No importer mapping exists for the Hugging Face model type '{modelType}', so importing it fails."),
            SupportLevel.Partial => new ModelCompatibility(CompatibilityState.Partial, modelType,
                $"The model type '{modelType}' is partially supported: {HfSupport.GetPartialSupportNote(modelType)}."),
            _ => new ModelCompatibility(CompatibilityState.Supported, modelType, $"The model type '{modelType}' is supported."),
        };
    }

    private static ModelCompatibility WithQuantization(string modelType, string? method, SupportLevel? level)
    {
        var architecture = ArchitectureOnly(modelType);
        return level switch
        {
            SupportLevel.Unsupported => new ModelCompatibility(CompatibilityState.UnsupportedWeightFormat, method,
                $"The weights are quantized with {method}, which the importer does not read, so importing fails. Unquantized, FP8 and gpt-oss MXFP4 checkpoints are read."),
            null => ModelCompatibility.Unknown($"{architecture.Reason} The weights are quantized with {method}, a method not known to this version; importing shows whether it is read."),
            _ => architecture with { Reason = architecture.Reason + (method is null ? " The weights are not quantized." : $" The weights are quantized with {method}, which is read.") },
        };
    }

    /// <summary>The first tensor type (in id order) that the importer does not know or that has no kernels.</summary>
    private static (string Name, int Tensors)? UnsupportedTensorType(IReadOnlyDictionary<int, int> tensorTypes)
    {
        foreach (var (id, tensors) in tensorTypes.OrderBy(pair => pair.Key))
        {
            if (GgufSupport.GetTensorType(id) is not { } type)
            {
                return ($"ggml type id {id}", tensors);
            }

            if (!CpuExecutionProvider.SupportsWeightType(type))
            {
                return (TypeName(type), tensors);
            }
        }

        return null;
    }

    private static string DescribeTypes(IReadOnlyDictionary<int, int> tensorTypes)
    {
        var listed = tensorTypes.OrderByDescending(pair => pair.Value).ThenBy(pair => pair.Key)
            .Select(pair => $"{(GgufSupport.GetTensorType(pair.Key) is { } type ? TypeName(type) : $"id {pair.Key}")} ({pair.Value})").ToList();
        return listed.Count <= ListedTypes ? string.Join(", ", listed) : string.Join(", ", listed.Take(ListedTypes)) + $" and {listed.Count - ListedTypes} more";
    }

    /// <summary>A data type as llama.cpp names it (e.g. <c>Q4_K</c>, <c>F32</c>).</summary>
    public static string TypeName(DataType type) => type.Name.ToUpperInvariant();

    private static string Count(int count, string noun) => count == 1 ? $"1 {noun} is" : $"{count} {noun}s are";

    private static string Shorten(string text) => text.Length <= ListedProblemCharacters ? text : text[..ListedProblemCharacters] + "…";
}
