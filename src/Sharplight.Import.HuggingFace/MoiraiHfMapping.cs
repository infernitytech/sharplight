using System.Globalization;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Salesforce Moirai 1.x (the uni2ts package's <c>MoiraiModule</c>: Salesforce/moirai-1.0-R-*, moirai-1.1-R-*) → canonical <c>moirai</c> (documented on
/// <c>MoiraiAdapter</c>). The checkpoint is not a transformers model: its config.json holds the module's keyword arguments (no <c>model_type</c>; it is
/// recognized by <c>distr_output</c> and <c>patch_sizes</c>, see <see cref="Detect"/>).
/// <para>
/// The multi-patch-size layers are split per patch size p: <c>in_proj.weight</c> [sizes, d, max p] becomes <c>in_proj.p{p}.weight</c> [d, p] (the columns
/// the module's mask keeps), and the distribution heads (one <c>MultiOutSizeLinear</c> per parameter) become one fused <c>param_proj.p{p}.weight</c>
/// [p · columns, d] whose rows are ordered (step, column): the mixture logits first, then every component's parameters in config order. The mixture logits keep
/// the module's layout, in which the logits of component c at step t are row c · max p + t of a head whose rows from components · p on are masked to
/// their bias. The binary variate bias (<c>var_attn_bias</c>) is kept; a univariate series adds the same bias to every score, so the graph does not read it.
/// </para>
/// </summary>
internal sealed class MoiraiHfMapping : IHfArchitectureMapping
{
    /// <summary>The model type of Moirai 1.x.</summary>
    public const string ModelType = "moirai";

    /// <inheritdoc/>
    public string Architecture => ModelType;

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = [ModelType];

    /// <inheritdoc/>
    public bool NeedsTokenizer => false;

    /// <summary>The model type of a uni2ts Moirai config (<c>moirai</c>, <see cref="Moirai2HfMapping.ModelType"/>), or <see langword="null"/>.</summary>
    public static string? Detect(HfConfig config)
    {
        var root = config.Root;
        if (root.ValueKind != JsonValueKind.Object || root.TryGetProperty("model_type", out _) || !root.TryGetProperty("d_model", out _))
        {
            return null;
        }

        if (root.TryGetProperty("distr_output", out var distr) && distr.ValueKind == JsonValueKind.Object && root.TryGetProperty("patch_sizes", out _))
        {
            return HasCentroids(config.Directory) ? MoiraiMoeHfMapping.ModelType : ModelType;
        }

        return root.TryGetProperty("quantile_levels", out _) && root.TryGetProperty("num_predict_token", out _) && root.TryGetProperty("patch_size", out _)
            ? Moirai2HfMapping.ModelType
            : null;
    }

    /// <summary>
    /// Whether the checkpoint in <paramref name="directory"/> is Moirai-MoE: its config has the same keys as Moirai 1.x, so the safetensors header is
    /// checked for the routing centroids (<c>encoder.centroid</c>).
    /// </summary>
    internal static bool HasCentroids(string directory)
    {
        string path = Path.Combine(directory, "model.safetensors");
        if (directory.Length == 0 || !File.Exists(path))
        {
            return false;
        }

        using var stream = File.OpenRead(path);
        Span<byte> length = stackalloc byte[8];
        if (stream.Read(length) != 8)
        {
            return false;
        }

        long size = BitConverter.ToInt64(length);
        if (size <= 0 || size > 100_000_000)
        {
            return false;
        }

        var header = new byte[size];
        stream.ReadExactly(header);
        return header.AsSpan().IndexOf("\"encoder.centroid\""u8) >= 0;
    }

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig c)
    {
        if (!TryComponents(c, out _, out string? error))
        {
            yield return error!;
        }

        if (c.GetBool("scaling") == false)
        {
            yield return "moirai: checkpoints without scaling (scaling = false) are not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        TryComponents(c, out var components, out _);
        int[] sizes = PatchSizes(c);
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = c.Int("d_model"),
            [a + "intermediate_size"] = c.GetInt("d_ff") ?? ((((4 * c.Int("d_model")) * 2 / 3) + 7) / 8 * 8),   // uni2ts' GLU default
            [a + "layer_count"] = c.Int("num_layers"),
            [a + "patch_sizes"] = string.Join(',', sizes),
            [a + "patch_size"] = sizes.Contains(32) ? 32 : sizes[^1],
            [a + "components"] = string.Join(',', components!.Select(x => x.Config)),
            [a + "max_sequence_length"] = c.GetInt("max_seq_len") ?? 512,
        };
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        TryComponents(c, out var components, out _);
        int[] sizes = PatchSizes(c);
        var collector = new HfWeightCollector();
        var heads = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
        WeightInfo? inWeight = null;
        WeightInfo? inBias = null;
        foreach (var (name, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            if (name.StartsWith("encoder.", StringComparison.Ordinal))
            {
                collector.Add(HfTransforms.Plain(info, name));
            }
            else if (name == "mask_encoding.weight")
            {
                collector.Add(HfTransforms.ToF32(info, name));
            }
            else if (name == "in_proj.weight")
            {
                inWeight = info;
            }
            else if (name == "in_proj.bias")
            {
                inBias = info;
            }
            else if (name.StartsWith("param_proj.proj.", StringComparison.Ordinal))
            {
                heads[name["param_proj.proj.".Length..]] = info;
            }
            else
            {
                collector.AddUnmapped(name);
            }
        }

        collector.ThrowIfUnmapped(ModelType);
        if (inWeight is null || inBias is null)
        {
            throw new InvalidDataException("moirai: the checkpoint has no in_proj.weight / in_proj.bias.");
        }

        int hidden = checked((int)inWeight.Shape[1].Fixed);
        AddPerSize(collector, "in_proj", inWeight, inBias, sizes, hidden, sizes.Max(), p => p);
        AddParamProj(collector, heads, components!, sizes, hidden);
        return collector.Weights;
    }

    /// <summary>
    /// Splits a multi-size layer (<c>MultiInSizeLinear</c> / <c>FeatLinear</c>: weight [sizes, d, max in], bias [sizes, d]) into <c>{name}.p{p}.weight</c>
    /// [d, in(p)] (the first in(p) columns, the ones the module's mask keeps) and <c>{name}.p{p}.bias</c> [d] per patch size.
    /// </summary>
    internal static void AddPerSize(HfWeightCollector collector, string name, WeightInfo weight, WeightInfo bias, int[] sizes, int hidden, int maxIn, Func<int, int> inputs)
    {
        for (int s = 0; s < sizes.Length; s++)
        {
            int index = s;
            int p = sizes[s];
            int width = inputs(p);
            collector.Add(WeightInfo.Derived($"{name}.p{p}.weight", DataType.F32, new Shape(hidden, width), () =>
            {
                var w = HfTransforms.DecodeF32(weight);   // [sizes, hidden, maxIn]
                var dst = new float[hidden * width];
                for (int o = 0; o < hidden; o++)
                {
                    Array.Copy(w, ((long)index * hidden * maxIn) + ((long)o * maxIn), dst, o * width, width);
                }

                return HfTransforms.EncodeF32(dst);
            }));
            collector.Add(WeightInfo.Derived($"{name}.p{p}.bias", DataType.F32, new Shape(hidden), () =>
                HfTransforms.EncodeF32(HfTransforms.DecodeF32(bias)[(index * hidden)..((index + 1) * hidden)])));
        }
    }

    /// <summary>The fused distribution head <c>param_proj.p{p}.{weight,bias}</c> of every patch size from the module's heads (names below <c>param_proj.proj.</c>).</summary>
    internal static void AddParamProj(HfWeightCollector collector, Dictionary<string, WeightInfo> heads, List<(string Config, string[] Parameters)> components, int[] sizes, int hidden)
    {
        // The heads in row order: the mixture logits, then each component's parameters (component c's leaves are named components.{c}.{parameter}).
        var leaves = new List<(WeightInfo Weight, WeightInfo Bias, int Width)>
        {
            (Head(heads, "weights_logits.weight"), Head(heads, "weights_logits.bias"), components.Count),
        };
        for (int k = 0; k < components.Count; k++)
        {
            foreach (string parameter in components[k].Parameters)
            {
                leaves.Add((Head(heads, $"components.{k}.{parameter}.weight"), Head(heads, $"components.{k}.{parameter}.bias"), 1));
            }
        }

        int maxSize = sizes.Max();
        int columns = leaves.Sum(l => l.Width);
        for (int s = 0; s < sizes.Length; s++)
        {
            int index = s;
            int p = sizes[s];
            collector.Add(WeightInfo.Derived($"param_proj.p{p}.weight", DataType.F32, new Shape(p * columns, hidden), () => FusedHead(leaves, index, p, maxSize, hidden, bias: false)));
            collector.Add(WeightInfo.Derived($"param_proj.p{p}.bias", DataType.F32, new Shape(p * columns), () => FusedHead(leaves, index, p, maxSize, hidden, bias: true)));
        }
    }

    internal static WeightInfo Head(Dictionary<string, WeightInfo> heads, string name)
        => heads.TryGetValue(name, out var w) ? w : throw new InvalidDataException($"moirai: the checkpoint has no param_proj.proj.{name}.");

    // Row (t, column) of the fused head of patch size p (index s): leaf l with width W (W columns per step) reads its row c · maxSize + t, masked to
    // zero weight from W · p on, as MultiOutSizeLinear does.
    internal static byte[] FusedHead(List<(WeightInfo Weight, WeightInfo Bias, int Width)> leaves, int s, int p, int maxSize, int hidden, bool bias)
    {
        int columns = leaves.Sum(l => l.Width);
        int stride = bias ? 1 : hidden;
        var dst = new float[p * columns * stride];
        int column = 0;
        foreach (var (weight, b, width) in leaves)
        {
            var src = HfTransforms.DecodeF32(bias ? b : weight);   // [sizes, width · maxSize (, hidden)]
            long rows = (long)width * maxSize;
            for (int c = 0; c < width; c++)
            {
                for (int t = 0; t < p; t++)
                {
                    long r = ((long)c * maxSize) + t;
                    int target = ((t * columns) + column + c) * stride;
                    if (bias)
                    {
                        dst[target] = src[((long)s * rows) + r];
                    }
                    else if (r < (long)width * p)
                    {
                        Array.Copy(src, (((long)s * rows) + r) * hidden, dst, target, hidden);
                    }
                }
            }

            column += width;
        }

        return HfTransforms.EncodeF32(dst);
    }

    internal static int[] PatchSizes(HfConfig c)
        => c.Root.TryGetProperty("patch_sizes", out var sizes) && sizes.ValueKind == JsonValueKind.Array ? [.. sizes.EnumerateArray().Select(x => x.GetInt32())] : [8, 16, 32, 64, 128];

    // The mixture components of distr_output (uni2ts.distribution.*Output classes) with their canonical names (see MoiraiAdapter) and parameter leaves.
    internal static bool TryComponents(HfConfig c, out List<(string Config, string[] Parameters)>? components, out string? error)
    {
        components = null;
        error = null;
        if (!c.Root.TryGetProperty("distr_output", out var distr) || distr.ValueKind != JsonValueKind.Object
            || !distr.TryGetProperty("_target_", out var target) || target.GetString() != "uni2ts.distribution.mixture.MixtureOutput"
            || !distr.TryGetProperty("components", out var list) || list.ValueKind != JsonValueKind.Array)
        {
            error = "moirai: only a MixtureOutput distr_output is supported.";
            return false;
        }

        components = [];
        foreach (var component in list.EnumerateArray())
        {
            string? name = component.TryGetProperty("_target_", out var t) ? t.GetString() : null;
            switch (name?[(name.LastIndexOf('.') + 1)..])
            {
                case "StudentTOutput":
                    components.Add(("student_t", ["df", "loc", "scale"]));
                    break;
                case "NormalOutput":
                    components.Add(("normal", ["loc", "scale"]));
                    break;
                case "NormalFixedScaleOutput":
                    double scale = component.TryGetProperty("scale", out var s) ? s.GetDouble() : 1e-3;
                    components.Add(("normal_fixed:" + scale.ToString("R", CultureInfo.InvariantCulture), ["loc"]));
                    break;
                case "NegativeBinomialOutput":
                    components.Add(("negative_binomial", ["total_count", "logits"]));
                    break;
                case "LogNormalOutput":
                    components.Add(("log_normal", ["loc", "scale"]));
                    break;
                default:
                    error = $"moirai: the mixture component {name} is not supported.";
                    components = null;
                    return false;
            }
        }

        return true;
    }
}
