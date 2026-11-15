using System.Globalization;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// What the Hugging Face importer accepts, asked before a checkpoint is downloaded (from its <c>config.json</c> and, for NVIDIA ModelOpt
/// checkpoints, <c>hf_quant_config.json</c>). Architecture answers come from the registered mappings; the quantization answer is the
/// one <see cref="HuggingFaceImporter.Open"/> applies.
/// </summary>
public static class HfSupport
{
    /// <summary>
    /// Methods whose weights the importer cannot read: their layouts (codebooks, trellis codes, rotations, framework-specific
    /// packings) have no conversion to a block type.
    /// </summary>
    private static readonly HashSet<string> s_unsupportedMethods = new(StringComparer.OrdinalIgnoreCase)
    {
        "hqq", "eetq", "quanto", "optimum-quanto", "exl2", "exllama", "torchao", "vptq", "higgs",
        "spqr", "fp_quant",
    };

    /// <summary>ModelOpt algorithms whose weights are read (<see cref="HfFp8Weights"/>, <see cref="HfFp4Weights"/>).</summary>
    private static readonly HashSet<string> s_modelOptAlgorithms = new(StringComparer.OrdinalIgnoreCase) { "FP8", "NVFP4", "MIXED_PRECISION" };

    /// <summary>The model type <c>mxfp4</c> checkpoints are read for (their expert blocks are mapped by that architecture).</summary>
    private const string Mxfp4ModelType = "gpt_oss";

    /// <summary>The compressed-tensors format name of MXFP4 packs (globally or per config group).</summary>
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
    /// Checkpoints of this model type are speculative-decoding drafters (EAGLE-3, Gemma 4 assistants): they are loaded as the draft source of the
    /// model they were trained for, not as models of their own.
    /// </summary>
    /// <param name="modelType">The <c>model_type</c> (see <see cref="GetModelType"/>).</param>
    public static bool IsDraftModel(string? modelType) => HfArchitectureMappings.Find(modelType)?.IsDraftModel == true;

    /// <summary>
    /// The checkpoint's quantization method and whether the importer reads it. Read: FP8 (<c>fp8</c> block scales, compressed-tensors
    /// with 8-bit float weights, ModelOpt <c>FP8</c>), NVFP4 (ModelOpt, compressed-tensors), MXFP4 (compressed-tensors; <c>mxfp4</c>
    /// for gpt-oss, AMD Quark), GPTQ (2, 3, 4 and 8 bits), AWQ (<c>gemm</c> / <c>gemv</c>), compressed-tensors integer weights,
    /// bitsandbytes, MLX quantization (affine and the <c>mxfp4</c> / <c>nvfp4</c> / <c>mxfp8</c> modes), AutoRound (GPTQ / AWQ packings)
    /// and AQLM (dequantized while read). Not read: the methods with codebook or framework-specific layouts (HQQ, EXL2, quanto, torchao, ...).
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

        // MLX writes {"group_size": 64, "bits": 4} without a method (as "quantization", and often also as "quantization_config").
        if (HfMlxWeights.Settings(config) is { } mlx)
        {
            string mode = String(mlx, "mode") ?? "affine";
            return new HfQuantizationSupport(mode == "affine" ? $"mlx ({mlx.GetProperty("bits").GetRawText()}-bit)" : $"mlx ({mode})",
                mode == "affine" || HfMlxFloatWeights.IsFloatMode(mode) ? SupportLevel.Supported : SupportLevel.Unsupported);
        }

        if (!config.TryGetQuantization("quantization_config", out var settings))
        {
            return config.TryGetQuantization("quantization", out var other)
                ? new HfQuantizationSupport($"mlx ({String(other, "mode") ?? "unknown"})", SupportLevel.Unsupported)
                : new HfQuantizationSupport(null, SupportLevel.Supported);
        }

        string? method = String(settings, "quant_method");
        if (method is null)
        {
            return new HfQuantizationSupport(settings.TryGetProperty("mode", out _) ? $"mlx ({String(settings, "mode")})" : null,
                settings.TryGetProperty("mode", out _) ? SupportLevel.Unsupported : null);
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
            case "gptq":
                return Gptq(method, settings);
            case "awq":
                return Awq(method, settings);
            case "auto-round" or "autoround" or "intel/auto-round":
                return AutoRound(settings);
            case "bitsandbytes":
                return new HfQuantizationSupport(method, SupportLevel.Supported);
            case "aqlm":
                // Dequantized to the codebook dtype while the weights are read (HfAqlmWeights).
                return new HfQuantizationSupport(method, SupportLevel.Supported);
            case "quark":
                return Quark(settings);
            default:
                return new HfQuantizationSupport(method, s_unsupportedMethods.Contains(method) ? SupportLevel.Unsupported : null);
        }
    }

    /// <summary>GPTQ: 2, 3, 4 and 8-bit codes in the <c>gptq</c> / <c>gptq_v2</c> layouts (<see cref="HfGptqLayouts"/>).</summary>
    private static HfQuantizationSupport Gptq(string method, JsonElement settings)
    {
        int bits = HfPackedBits.Int(settings, "bits", 4);
        string format = String(settings, "checkpoint_format") ?? "gptq";
        bool read = bits is 2 or 3 or 4 or 8 && format is "gptq" or "gptq_v2";
        return new HfQuantizationSupport(string.Create(CultureInfo.InvariantCulture, $"{method} ({bits}-bit)"), read ? SupportLevel.Supported : SupportLevel.Unsupported);
    }

    /// <summary>AutoRound: integer weights packed like GPTQ or AWQ GEMM (<see cref="HfAutoRound"/>).</summary>
    private static HfQuantizationSupport AutoRound(JsonElement settings)
    {
        string? layout = HfAutoRound.Layout(settings);
        int bits = HfPackedBits.Int(settings, "bits", 4);
        bool read = layout switch
        {
            "gptq" => bits is 2 or 3 or 4 or 8,
            "awq" => bits == 4,
            _ => false,
        };
        return new HfQuantizationSupport(string.Create(CultureInfo.InvariantCulture, $"auto-round ({layout ?? String(settings, "packing_format") ?? "unknown"}, {bits}-bit)"),
            read ? SupportLevel.Supported : SupportLevel.Unsupported);
    }

    /// <summary>AWQ: 4-bit codes in the <c>gemm</c> and <c>gemv</c> layouts (<see cref="HfAwqLayouts"/>).</summary>
    private static HfQuantizationSupport Awq(string method, JsonElement settings)
    {
        int bits = HfPackedBits.Int(settings, "bits", HfPackedBits.Int(settings, "w_bit", 4));
        string version = (String(settings, "version") ?? "gemm").ToLowerInvariant();
        bool read = bits == 4 && version is "gemm" or "gemv";
        return new HfQuantizationSupport($"{method} ({version})", read ? SupportLevel.Supported : SupportLevel.Unsupported);
    }

    /// <summary>
    /// compressed-tensors: 8-bit float weights (<see cref="HfFp8Weights"/>), 4-bit float weights (NVFP4 / MXFP4,
    /// <see cref="HfFp4Weights"/>) and 1 to 8-bit integer weights (<see cref="HfCompressedIntLayouts"/>) are read; sparse
    /// (bitmask-compressed) checkpoints are not.
    /// </summary>
    private static HfQuantizationSupport CompressedTensors(JsonElement settings)
    {
        bool mxfp4Pack = String(settings, "format") == Mxfp4PackFormat;
        if (!settings.TryGetProperty("config_groups", out var groups) || groups.ValueKind != JsonValueKind.Object)
        {
            return new HfQuantizationSupport("compressed-tensors", null);
        }

        string? label = null;
        foreach (var group in groups.EnumerateObject())
        {
            if (group.Value.ValueKind != JsonValueKind.Object || !group.Value.TryGetProperty("weights", out var weights)
                || weights.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            string type = String(weights, "type") ?? "int";
            int bits = weights.TryGetProperty("num_bits", out var value) && value.ValueKind == JsonValueKind.Number ? value.GetInt32() : 0;
            bool isFloat = string.Equals(type, "float", StringComparison.OrdinalIgnoreCase);
            bool read = isFloat ? bits is 4 or 8 : bits is >= 1 and <= 8;
            bool mxfp4Packed = isFloat && bits == 4 && (mxfp4Pack || String(group.Value, "format") == Mxfp4PackFormat);
            label = mxfp4Packed ? "compressed-tensors (mxfp4)" : string.Create(CultureInfo.InvariantCulture, $"compressed-tensors ({type} {bits}-bit)");
            if (!read)
            {
                return new HfQuantizationSupport(label, SupportLevel.Unsupported);
            }
        }

        if (settings.TryGetProperty("sparsity_config", out var sparsity) && sparsity.ValueKind == JsonValueKind.Object
            && String(sparsity, "format") is { } sparseFormat && sparseFormat != "dense")
        {
            return new HfQuantizationSupport($"compressed-tensors ({sparseFormat})", SupportLevel.Unsupported);
        }

        return new HfQuantizationSupport(label ?? "compressed-tensors", SupportLevel.Supported);
    }

    /// <summary>
    /// AMD Quark: weights quantized to OCP MXFP4 (<c>fp4</c>, 32-value groups, <c>e8m0</c> scales: U8 <c>X.weight</c> and
    /// <c>X.weight_scale</c>, low nibble first, read by <see cref="HfFp4Weights"/>) or to FP8 E4M3 (<see cref="HfFp8Weights"/>), in
    /// the global and every per-layer spec. Activation specs are ignored (activations are not quantized). Other weight types
    /// (int4 / int8 in Quark's reordered packing, MXFP6, lists of specs) are not read.
    /// </summary>
    private static HfQuantizationSupport Quark(JsonElement settings)
    {
        var specs = new List<JsonElement>();
        if (settings.TryGetProperty("global_quant_config", out var global) && global.ValueKind == JsonValueKind.Object)
        {
            specs.Add(global);
        }

        foreach (string name in (string[])["layer_quant_config", "layer_type_quant_config"])
        {
            if (settings.TryGetProperty(name, out var layers) && layers.ValueKind == JsonValueKind.Object)
            {
                specs.AddRange(layers.EnumerateObject().Select(layer => layer.Value));
            }
        }

        var types = new SortedSet<string>(StringComparer.Ordinal);
        foreach (var spec in specs)
        {
            if (spec.ValueKind != JsonValueKind.Object || !spec.TryGetProperty("weight", out var weight) || weight.ValueKind == JsonValueKind.Null)
            {
                continue;
            }

            string? type = String(weight, "dtype");
            bool mxfp4 = type == "fp4" && HfPackedBits.Int(weight, "group_size", 0) == 32 && String(weight, "scale_format") == "e8m0";
            if (!mxfp4 && type != "fp8_e4m3")
            {
                return new HfQuantizationSupport($"quark ({type ?? "weight list"})", SupportLevel.Unsupported);
            }

            types.Add(mxfp4 ? "mxfp4" : "fp8");
        }

        return types.Count == 0
            ? new HfQuantizationSupport("quark", null)
            : new HfQuantizationSupport($"quark ({string.Join(", ", types)})", SupportLevel.Supported);
    }

    private static HfQuantizationSupport ModelOptAlgorithm(string? algorithm)
    {
        string label = algorithm is null ? "modelopt" : $"modelopt {algorithm}";
        return algorithm is null
            ? new HfQuantizationSupport(label, null)
            : new HfQuantizationSupport(label, s_modelOptAlgorithms.Contains(algorithm) ? SupportLevel.Supported : SupportLevel.Unsupported);
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

    /// <summary>
    /// The reasons importing a checkpoint fails, from its <c>config.json</c> alone (before the checkpoint is downloaded): no mapping for
    /// its model type, the mapping's feature checks, and its hyper-parameter mapping. Tensors are not read. Empty when the configuration
    /// maps; the quantization is assessed separately (<see cref="GetQuantizationSupport"/>).
    /// </summary>
    /// <param name="configJson">The text of <c>config.json</c>.</param>
    public static IReadOnlyList<string> Validate(string configJson)
    {
        using var config = TryParse(configJson);
        if (config is null)
        {
            return ["config.json is not a JSON object."];
        }

        if (HfArchitectureMappings.Find(config.ModelType) is not { } mapping)
        {
            return [$"Hugging Face model_type '{config.ModelType}' has no mapping."];
        }

        try
        {
            if (mapping.Validate(config).ToList() is { Count: > 0 } problems)
            {
                return problems;
            }

            mapping.MapHyperParameters(config, HuggingFaceImportOptions.Default);
            return [];
        }
        catch (Exception error) when (error is NotSupportedException or InvalidDataException or KeyNotFoundException or FormatException
            or InvalidOperationException or InvalidCastException or JsonException)
        {
            return [error.Message];
        }
    }
}
