using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// The activation scheme (<see cref="WeightInfo.Activations"/>) a checkpoint's quantization config declares for each quantized module:
/// weight-only (W4A16, W8A16) or activations quantized too (W8A8, W4A4, static or dynamic input scales).
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>NVIDIA ModelOpt (<c>hf_quant_config.json</c>, or <c>quantization_config</c> with <c>quant_method modelopt</c>): the
/// <c>quant_algo</c> of the module (<c>quantized_layers</c> of <c>MIXED_PRECISION</c>) or of the checkpoint. Algorithms with
/// <c>A16</c> or the suffix <c>_WO</c> (<c>W4A16_NVFP4</c>, <c>W4A16_AWQ</c>, <c>FP8_PB_WO</c>) and <c>INT4_AWQ</c> are weight-only;
/// <c>FP8*</c>, <c>NVFP4*</c>, <c>MXFP8</c>, <c>MXFP4</c>, <c>W4A8*</c> and <c>INT8_SQ</c> quantize the activations.</item>
/// <item>compressed-tensors schemes (<c>config_groups</c>, also in ModelOpt's exported config): a group with
/// <c>input_activations</c> quantizes them, one without is weight-only; groups name their modules by name, by <c>re:</c> pattern or by
/// class (<c>Linear</c>, every module).</item>
/// <item><c>quant_method</c> <c>fp8</c>, <c>mxfp8</c> and <c>fbgemm_fp8</c> quantize the activations (dynamically or with the
/// declared <c>activation_scheme</c>); GPTQ, AWQ, AutoRound, HQQ, EETQ, AQLM, VPTQ, HIGGS and EXL2 are weight-only; optimum-quanto
/// is weight-only without <c>activations</c>; torchao by its <c>quant_type</c> configs (<c>...WeightOnlyConfig</c> /
/// <c>...DynamicActivation...</c>).</item>
/// </list>
/// Anything else (unknown algorithms, ggml block groups, formats with their own activation handling) stays
/// <see cref="ActivationScheme.Unspecified"/>.
/// </remarks>
internal sealed class HfActivationSchemes
{
    private static readonly HashSet<string> s_weightOnlyMethods = new(StringComparer.OrdinalIgnoreCase)
    {
        "gptq", "awq", "auto-round", "autoround", "intel/auto-round", "hqq", "eetq", "aqlm", "vptq", "higgs", "exl2",
    };

    private static readonly HashSet<string> s_activationMethods = new(StringComparer.OrdinalIgnoreCase) { "fp8", "mxfp8", "fbgemm_fp8" };

    /// <summary>ModelOpt algorithms (by prefix) whose products quantize the activations.</summary>
    private static readonly string[] s_activationAlgorithmPrefixes = ["FP8", "NVFP4", "MXFP8", "MXFP4", "W4A8", "INT8_SQ"];

    private readonly ActivationScheme _default;
    private readonly List<(string Module, ActivationScheme Scheme)> _modules;
    private readonly List<(Regex Pattern, ActivationScheme Scheme)> _patterns;

    private HfActivationSchemes(ActivationScheme fallback, List<(string, ActivationScheme)>? modules = null, List<(Regex, ActivationScheme)>? patterns = null)
    {
        _default = fallback;
        _modules = modules ?? [];
        _patterns = patterns ?? [];
    }

    /// <summary>The schemes of a checkpoint.</summary>
    /// <param name="config">The checkpoint config (its <c>quantization_config</c>).</param>
    /// <param name="modelOpt">The <c>quantization</c> object of <c>hf_quant_config.json</c>, when the checkpoint has one.</param>
    public static HfActivationSchemes From(HfConfig config, JsonElement? modelOpt)
    {
        if (modelOpt is { ValueKind: JsonValueKind.Object } producer)
        {
            return FromModelOpt(producer);
        }

        if (!config.TryGetQuantization("quantization_config", out var settings))
        {
            return new HfActivationSchemes(ActivationScheme.Unspecified);
        }

        if (settings.TryGetProperty("config_groups", out var groups) && groups.ValueKind == JsonValueKind.Object)
        {
            return FromGroups(groups);
        }

        string method = HfPackedBits.String(settings, "quant_method") ?? "";
        if (method.StartsWith("modelopt", StringComparison.OrdinalIgnoreCase))
        {
            return FromModelOpt(settings);
        }

        return new HfActivationSchemes(MethodScheme(method, settings));
    }

    /// <summary>The scheme of the checkpoint weight <paramref name="weightName"/> (a module name with a parameter suffix).</summary>
    public ActivationScheme For(string weightName)
    {
        string module = weightName.EndsWith(".weight", StringComparison.Ordinal) ? weightName[..^".weight".Length] : weightName;
        int best = -1;
        var scheme = ActivationScheme.Unspecified;
        foreach (var (name, s) in _modules)
        {
            bool covers = module == name || (module.StartsWith(name, StringComparison.Ordinal) && module[name.Length] == '.');
            if (covers && name.Length > best)
            {
                best = name.Length;
                scheme = s;
            }
        }

        if (best >= 0)
        {
            return scheme;
        }

        foreach (var (pattern, s) in _patterns)
        {
            if (pattern.IsMatch(module))
            {
                return s;
            }
        }

        return _default;
    }

    /// <summary>The scheme of a ModelOpt algorithm name.</summary>
    internal static ActivationScheme Algorithm(string? algorithm)
    {
        if (algorithm is null)
        {
            return ActivationScheme.Unspecified;
        }

        string a = algorithm.ToUpperInvariant();
        if (a.Contains("A16", StringComparison.Ordinal) || a.EndsWith("_WO", StringComparison.Ordinal) || a == "INT4_AWQ")
        {
            return ActivationScheme.Unquantized;
        }

        return s_activationAlgorithmPrefixes.Any(p => a.StartsWith(p, StringComparison.Ordinal)) ? ActivationScheme.Quantized : ActivationScheme.Unspecified;
    }

    /// <summary>ModelOpt: the checkpoint's <c>quant_algo</c>, and per module the <c>quantized_layers</c> entries.</summary>
    private static HfActivationSchemes FromModelOpt(JsonElement quantization)
    {
        var modules = new List<(string, ActivationScheme)>();
        if (quantization.TryGetProperty("quantized_layers", out var layers) && layers.ValueKind == JsonValueKind.Object)
        {
            foreach (var layer in layers.EnumerateObject())
            {
                if (layer.Value.ValueKind == JsonValueKind.Object)
                {
                    modules.Add((layer.Name, Algorithm(HfPackedBits.String(layer.Value, "quant_algo"))));
                }
            }
        }

        return new HfActivationSchemes(Algorithm(HfPackedBits.String(quantization, "quant_algo")), modules);
    }

    /// <summary>compressed-tensors schemes: per group, quantized activations when it declares <c>input_activations</c>.</summary>
    private static HfActivationSchemes FromGroups(JsonElement groups)
    {
        var schemes = new List<(JsonElement Targets, ActivationScheme Scheme)>();
        foreach (var group in groups.EnumerateObject())
        {
            var g = group.Value;
            if (g.ValueKind != JsonValueKind.Object)
            {
                continue;
            }

            var scheme = g.TryGetProperty("weights", out var weights) && weights.ValueKind == JsonValueKind.Object
                ? g.TryGetProperty("input_activations", out var input) && input.ValueKind == JsonValueKind.Object
                    ? ActivationScheme.Quantized
                    : ActivationScheme.Unquantized
                : Algorithm(HfPackedBits.String(g, "quant_algo"));
            if (g.TryGetProperty("targets", out var targets) && targets.ValueKind == JsonValueKind.Array)
            {
                schemes.Add((targets, scheme));
            }
        }

        var modules = new List<(string, ActivationScheme)>();
        var patterns = new List<(Regex, ActivationScheme)>();
        var classTargets = new List<ActivationScheme>();
        foreach (var (targets, scheme) in schemes)
        {
            foreach (var target in targets.EnumerateArray())
            {
                string? t = target.ValueKind == JsonValueKind.String ? target.GetString() : null;
                if (string.IsNullOrEmpty(t))
                {
                    continue;
                }

                if (t.StartsWith("re:", StringComparison.Ordinal))
                {
                    // compressed-tensors matches with Python's re.match: anchored at the start of the module name.
                    patterns.Add((new Regex(@"\A(?:" + t[3..] + ")", RegexOptions.CultureInvariant), scheme));
                }
                else if (!t.Contains('.', StringComparison.Ordinal) && char.IsUpper(t[0]))
                {
                    // A module class (Linear): every quantized module of the group.
                    classTargets.Add(scheme);
                }
                else
                {
                    modules.Add((t, scheme));
                }
            }
        }

        // Class targets cover the modules no name or pattern names; groups of different schemes over one class leave them unspecified.
        var fallback = classTargets.Count > 0 && classTargets.All(c => c == classTargets[0]) ? classTargets[0] : ActivationScheme.Unspecified;
        return new HfActivationSchemes(fallback, modules, patterns);
    }

    /// <summary>The scheme of a checkpoint-wide <c>quant_method</c>.</summary>
    private static ActivationScheme MethodScheme(string method, JsonElement settings)
    {
        if (s_activationMethods.Contains(method))
        {
            return ActivationScheme.Quantized;
        }

        if (s_weightOnlyMethods.Contains(method))
        {
            return ActivationScheme.Unquantized;
        }

        if (method is "quanto" or "optimum-quanto")
        {
            return settings.TryGetProperty("activations", out var activations) && activations.ValueKind == JsonValueKind.String
                ? ActivationScheme.Quantized
                : ActivationScheme.Unquantized;
        }

        return method == "torchao" ? TorchAo(settings) : ActivationScheme.Unspecified;
    }

    /// <summary>torchao: the scheme all <c>quant_type</c> configs agree on (weight-only or dynamic activation), else unspecified.</summary>
    private static ActivationScheme TorchAo(JsonElement settings)
    {
        if (!settings.TryGetProperty("quant_type", out var quantType) || quantType.ValueKind != JsonValueKind.Object)
        {
            return ActivationScheme.Unspecified;
        }

        var found = new HashSet<ActivationScheme>();
        foreach (var entry in quantType.EnumerateObject())
        {
            string type = (entry.Value.ValueKind == JsonValueKind.Object ? HfPackedBits.String(entry.Value, "_type") : null) ?? "";
            found.Add(type.Contains("WeightOnly", StringComparison.Ordinal) ? ActivationScheme.Unquantized
                : type.Contains("DynamicActivation", StringComparison.Ordinal) ? ActivationScheme.Quantized : ActivationScheme.Unspecified);
        }

        return found.Count == 1 ? found.First() : ActivationScheme.Unspecified;
    }
}
