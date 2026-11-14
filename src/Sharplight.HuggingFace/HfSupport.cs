using System.Globalization;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// What the Hugging Face importer accepts, asked before a checkpoint is downloaded (from its <c>config.json</c> and, for NVIDIA ModelOpt
/// checkpoints, <c>hf_quant_config.json</c>). Architecture answers come from the registered mappings; the quantization answer is the
/// one <see cref="HuggingFaceImporter.Open"/> applies.
/// </summary>
public static class HfSupport
{
    /// <summary>
    /// Methods whose weights the importer cannot read: packed integer formats (GPTQ, AWQ, bitsandbytes, MLX, ...) need dequantization
    /// kernels for their layouts, which do not exist.
    /// </summary>
    private static readonly HashSet<string> s_unsupportedMethods = new(StringComparer.OrdinalIgnoreCase)
    {
        "gptq", "awq", "bitsandbytes", "aqlm", "hqq", "eetq", "quanto", "optimum-quanto", "exl2", "exllama", "torchao", "vptq", "higgs",
        "spqr", "auto-round", "autoround", "fp_quant", "quark", "gptqmodel",
    };

    /// <summary>The model type <c>mxfp4</c> checkpoints are read for (their expert blocks are mapped by that architecture).</summary>
    private const string Mxfp4ModelType = "gpt_oss";
    private const string Mxfp4PackFormat = "mxfp4-pack-quantized";

    /// <summary>Every <c>model_type</c> the importer maps, in no particular order.</summary>
    public static IReadOnlyCollection<string> ModelTypes => [.. HfArchitectureMappings.All.SelectMany(mapping => mapping.ModelTypes)];

    /// <summary>
    /// The model type the importer selects a mapping by: <c>model_type</c>, else that of <c>text_config</c>, else one implied by a known
    /// remote-code <c>architectures</c> class; EAGLE-3 drafters are <c>eagle3</c>. Null when the text is not a JSON object or names none.
    /// </summary>
    /// <param name="configJson">The text of <c>config.json</c>.</param>
    public static string? GetModelType(string configJson)
    {
        using var config = TryParse(configJson);
        return config?.ModelType;
    }

    /// <summary>How far a model type is supported: unsupported without a mapping, partial when its mapping names a restriction.</summary>
    /// <param name="modelType">The <c>model_type</c> (see <see cref="GetModelType"/>).</param>
    public static SupportLevel GetArchitectureSupport(string? modelType)
    {
        if (HfArchitectureMappings.Find(modelType) is not { } mapping)
        {
            return SupportLevel.Unsupported;
        }

        return mapping.PartialSupport(modelType!) is null ? SupportLevel.Supported : SupportLevel.Partial;
    }

    /// <summary>What restricts a partially supported model type, or null when it is fully supported or not supported at all.</summary>
    /// <param name="modelType">The <c>model_type</c>.</param>
    public static string? GetPartialSupportNote(string? modelType) => HfArchitectureMappings.Find(modelType)?.PartialSupport(modelType!);

    /// <summary>
    /// The checkpoint's quantization method and whether the importer reads it. Read: FP8 (<c>fp8</c> block scales, compressed-tensors
    /// with 8-bit float weights, ModelOpt <c>FP8</c>) and <c>mxfp4</c> for gpt-oss. Not read: integer and 4-bit packed formats (GPTQ, AWQ,
    /// bitsandbytes, MLX, compressed-tensors with integer or 4-bit weights, ModelOpt NVFP4, ...).
    /// </summary>
    /// <param name="configJson">The text of <c>config.json</c> (its <c>quantization_config</c>).</param>
    /// <param name="quantizationConfigJson">The text of <c>hf_quant_config.json</c> when the checkpoint has one (ModelOpt), else null.</param>
    public static HfQuantizationSupport GetQuantizationSupport(string configJson, string? quantizationConfigJson = null)
    {
        using var config = TryParse(configJson);
        if (config is null)
        {
            return new HfQuantizationSupport(null, null);
        }

        using var modelOpt = quantizationConfigJson is null ? null : TryParseDocument(quantizationConfigJson);
        return AssessQuantization(config, modelOpt?.RootElement);
    }

    /// <summary>The quantization of a parsed config (see the public overload).</summary>
    /// <param name="config">The config.</param>
    /// <param name="modelOpt">The root of <c>hf_quant_config.json</c>, when present.</param>
    internal static HfQuantizationSupport AssessQuantization(HfConfig config, JsonElement? modelOpt)
    {
        if (modelOpt is { ValueKind: JsonValueKind.Object } root && root.TryGetProperty("quantization", out var quantization)
            && quantization.ValueKind == JsonValueKind.Object)
        {
            return ModelOptAlgorithm(String(quantization, "quant_algo"));
        }

        if (!config.Root.TryGetProperty("quantization_config", out var settings) || settings.ValueKind != JsonValueKind.Object)
        {
            return new HfQuantizationSupport(null, SupportLevel.Supported);
        }

        string? method = String(settings, "quant_method");
        if (method is null)
        {
            // MLX writes {"group_size": 64, "bits": 4} without a method.
            return settings.TryGetProperty("bits", out var bits) && bits.ValueKind == JsonValueKind.Number
                ? new HfQuantizationSupport($"mlx ({bits.GetRawText()}-bit)", SupportLevel.Unsupported)
                : new HfQuantizationSupport(null, null);
        }

        switch (method.ToLowerInvariant())
        {
            case "fp8":
                return new HfQuantizationSupport(method, SupportLevel.Supported);
            case "mxfp4":
                return new HfQuantizationSupport(method,
                    string.Equals(config.ModelType, Mxfp4ModelType, StringComparison.OrdinalIgnoreCase) ? SupportLevel.Supported : SupportLevel.Unsupported);
            case "compressed-tensors":
                return CompressedTensors(settings);
            case "modelopt":
                return ModelOptAlgorithm(String(settings, "quant_algo"));
            default:
                return new HfQuantizationSupport(method, s_unsupportedMethods.Contains(method) ? SupportLevel.Unsupported : null);
        }
    }

    /// <summary>
    /// compressed-tensors: 8-bit float weights are read (<see cref="HfFp8Weights"/>), as are 4-bit float groups in the MXFP4 pack format
    /// (<see cref="HfMxfp4Weights"/>); integer or other narrower ones are not.
    /// </summary>
    private static HfQuantizationSupport CompressedTensors(JsonElement settings)
    {
        bool mxfp4Pack = String(settings, "format") == Mxfp4PackFormat;
        bool anyPacked = mxfp4Pack;
        if (!settings.TryGetProperty("config_groups", out var groups) || groups.ValueKind != JsonValueKind.Object)
        {
            return new HfQuantizationSupport("compressed-tensors", null);
        }

        foreach (var group in groups.EnumerateObject())
        {
            if (group.Value.ValueKind != JsonValueKind.Object || !group.Value.TryGetProperty("weights", out var weights)
                || weights.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            string type = String(weights, "type") ?? "int";
            int bits = weights.TryGetProperty("num_bits", out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : 0;
            bool packed = mxfp4Pack || String(group.Value, "format") == Mxfp4PackFormat;
            anyPacked |= packed && bits == 4;
            if (!string.Equals(type, "float", StringComparison.OrdinalIgnoreCase) || !(bits == 8 || (bits == 4 && packed)))
            {
                string label = string.Create(CultureInfo.InvariantCulture, $"compressed-tensors ({type} {bits}-bit)");
                return new HfQuantizationSupport(label, SupportLevel.Unsupported);
            }
        }

        return new HfQuantizationSupport(anyPacked ? "compressed-tensors (mxfp4)" : "compressed-tensors (float 8-bit)", SupportLevel.Supported);
    }

    private static HfQuantizationSupport ModelOptAlgorithm(string? algorithm)
    {
        string label = algorithm is null ? "modelopt" : $"modelopt {algorithm}";
        return algorithm is null
            ? new HfQuantizationSupport(label, null)
            : new HfQuantizationSupport(label, string.Equals(algorithm, "FP8", StringComparison.OrdinalIgnoreCase) ? SupportLevel.Supported : SupportLevel.Unsupported);
    }

    private static HfConfig? TryParse(string json)
    {
        try
        {
            var config = HfConfig.Parse(json, "");
            if (config.Root.ValueKind == JsonValueKind.Object)
            {
                return config;
            }

            config.Dispose();
            return null;
        }
        catch (Exception error) when (error is JsonException or InvalidOperationException)
        {
            return null;
        }
    }

    private static JsonDocument? TryParseDocument(string json)
    {
        try
        {
            return JsonDocument.Parse(json, new JsonDocumentOptions { AllowTrailingCommas = true });
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static string? String(JsonElement element, string name)
        => element.TryGetProperty(name, out var value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
}
