using System.Text.RegularExpressions;
using Sharplight.Models;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>gpt2</c> (GPT-2, DistilGPT-2, ...): learned absolute position embeddings (no rotary), layer norm with bias, fused q/k/v and plain GELU MLP.
/// The checkpoint's projections are transposed <c>Conv1D</c> weights ([in, out]); they are transposed lazily into the canonical [out, in] layout.
/// </summary>
internal sealed partial class Gpt2HfMapping : HfDecoderFamilyMapping
{
    private static readonly IReadOnlyDictionary<string, string> s_layerTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["attn.c_proj.weight"] = "attention.o.weight",
        ["attn.c_proj.bias"] = "attention.o.bias",
        ["mlp.c_fc.weight"] = "mlp.up.weight",
        ["mlp.c_fc.bias"] = "mlp.up.bias",
        ["mlp.c_proj.weight"] = "mlp.down.weight",
        ["mlp.c_proj.bias"] = "mlp.down.bias",
        ["ln_1.weight"] = "input_norm.weight",
        ["ln_1.bias"] = "input_norm.bias",
        ["ln_2.weight"] = "post_attention_norm.weight",
        ["ln_2.bias"] = "post_attention_norm.bias",
    };

    private static readonly IReadOnlyDictionary<string, string> s_globalTensors = new Dictionary<string, string>(StringComparer.Ordinal)
    {
        ["transformer.wte.weight"] = "token_embedding.weight",
        ["transformer.wpe.weight"] = "position_embedding.weight",
        ["transformer.ln_f.weight"] = "output_norm.weight",
        ["transformer.ln_f.bias"] = "output_norm.bias",
        ["wte.weight"] = "token_embedding.weight",
        ["wpe.weight"] = "position_embedding.weight",
        ["ln_f.weight"] = "output_norm.weight",
        ["ln_f.bias"] = "output_norm.bias",
        ["lm_head.weight"] = "lm_head.weight",
    };

    /// <summary>Weights stored as <c>Conv1D</c> ([in, out]) that become linear weights ([out, in]).</summary>
    private static readonly HashSet<string> s_transposed = new(StringComparer.Ordinal)
    {
        "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight",
    };

    [GeneratedRegex(@"^(?:transformer\.)?h\.(\d+)\.(.+)$")]
    private static partial Regex Gpt2LayerName();

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["gpt2"];

    /// <inheritdoc/>
    protected override bool LayerNorm => true;

    /// <inheritdoc/>
    protected override bool Gated => false;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> LayerTensors => s_layerTensors;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string> GlobalTensors => s_globalTensors;

    /// <inheritdoc/>
    protected override Regex LayerName => Gpt2LayerName();

    /// <inheritdoc/>
    protected override int HiddenSize(HfConfig c) => c.Int("n_embd", "hidden_size");

    /// <inheritdoc/>
    protected override int HeadCount(HfConfig c) => c.Int("n_head", "num_attention_heads");

    /// <inheritdoc/>
    protected override int LayerCount(HfConfig c) => c.Int("n_layer", "num_hidden_layers");

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.GetInt("n_inner") ?? 4 * HiddenSize(c);

    /// <inheritdoc/>
    protected override int KvHeadCount(HfConfig c, int headCount) => headCount;

    /// <inheritdoc/>
    protected override int? ContextLength(HfConfig c) => c.GetInt("n_positions", "max_position_embeddings");

    /// <inheritdoc/>
    protected override bool IsIgnored(string name)
        => base.IsIgnored(name) || name.EndsWith(".attn.bias", StringComparison.Ordinal) || name.EndsWith(".attn.masked_bias", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        md["decoder.rope.layers"] = new int[LayerCount(c)];
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        if (suffix == "attn.c_attn.weight")
        {
            var linear = Transpose(w, layerPrefix + "attn.qkv.weight");
            int hidden = HiddenSize(c);
            collector.Add(linear.RowSlice(layerPrefix + "attention.q.weight", 0, hidden));
            collector.Add(linear.RowSlice(layerPrefix + "attention.k.weight", hidden, hidden));
            collector.Add(linear.RowSlice(layerPrefix + "attention.v.weight", 2 * hidden, hidden));
            return true;
        }

        if (suffix == "attn.c_attn.bias")
        {
            int hidden = HiddenSize(c);
            collector.Add(HfTransforms.Plain(w.RowSlice("", 0, hidden), layerPrefix + "attention.q.bias"));
            collector.Add(HfTransforms.Plain(w.RowSlice("", hidden, hidden), layerPrefix + "attention.k.bias"));
            collector.Add(HfTransforms.Plain(w.RowSlice("", 2 * hidden, hidden), layerPrefix + "attention.v.bias"));
            return true;
        }

        if (!s_transposed.Contains(suffix))
        {
            return false;
        }

        collector.Add(Transpose(w, layerPrefix + s_layerTensors[suffix]));
        return true;
    }

    /// <summary>The lazily transposed copy of a rank-2 weight stored as [in, out].</summary>
    private static WeightInfo Transpose(WeightInfo w, string name)
    {
        long rows = w.Shape[0].Fixed;
        long columns = w.Shape[1].Fixed;
        int elementBytes = checked((int)w.Type.ByteSize(1));
        return WeightInfo.Derived(name, w.Type, new Sharplight.Tensors.Shape(columns, rows), () =>
        {
            var source = w.ReadAll();
            var result = new byte[source.Length];
            for (long r = 0; r < rows; r++)
            {
                for (long col = 0; col < columns; col++)
                {
                    Buffer.BlockCopy(source, checked((int)((r * columns + col) * elementBytes)), result, checked((int)((col * rows + r) * elementBytes)), elementBytes);
                }
            }

            return result;
        });
    }
}
