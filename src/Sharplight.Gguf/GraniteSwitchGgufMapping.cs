using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// llama.cpp <c>graniteswitch</c> (Granite Switch: dense Granite with per-token switched LoRA adapters). The base layers are Granite's
/// (muP multipliers, fused <c>attn_qkv</c> split, q/k rows un-permuted); every q/k/v/o/gate/up/down projection carries its stacked adapter pairs
/// <c>{tensor}.lora_a</c> / <c>{tensor}.lora_b</c> (slot 0 the zero adapter the converter adds; the q/k <c>lora_b</c> rows are un-permuted like
/// the base rows), and the <c>adapters.*</c> metadata (activation and substitute tokens, router gain) becomes the canonical adapter token table
/// (<see cref="AdapterTokenTable"/>).
/// </summary>
internal sealed class GraniteSwitchGgufMapping : GgufDecoderFamilyMapping
{
    private static readonly Dictionary<string, string> s_layerTensors = new(StringComparer.Ordinal)
    {
        ["attn_q.lora_a"] = "attention.q.lora_a",
        ["attn_q.lora_b"] = "attention.q.lora_b",
        ["attn_k.lora_a"] = "attention.k.lora_a",
        ["attn_k.lora_b"] = "attention.k.lora_b",
        ["attn_v.lora_a"] = "attention.v.lora_a",
        ["attn_v.lora_b"] = "attention.v.lora_b",
        ["attn_output.lora_a"] = "attention.o.lora_a",
        ["attn_output.lora_b"] = "attention.o.lora_b",
        ["ffn_gate.lora_a"] = "mlp.gate.lora_a",
        ["ffn_gate.lora_b"] = "mlp.gate.lora_b",
        ["ffn_up.lora_a"] = "mlp.up.lora_a",
        ["ffn_up.lora_b"] = "mlp.up.lora_b",
        ["ffn_down.lora_a"] = "mlp.down.lora_a",
        ["ffn_down.lora_b"] = "mlp.down.lora_b",
    };

    /// <inheritdoc/>
    public override IReadOnlyList<string> GgufArchitectures { get; } = ["graniteswitch"];

    /// <inheritdoc/>
    protected override bool PermutesQk => true;

    /// <inheritdoc/>
    protected override IReadOnlyDictionary<string, string>? LayerTensorOverrides => s_layerTensors;

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, GgufMetadataView g)
    {
        GgufMupScales.Add(md, g);
        if (!g.Bool("rope.scaling.finetuned", true))
        {
            md[GgufDecoderHyperParameters.Prefix + "rope.layers"] = new int[g.Int("block_count")];
        }

        md[GgufDecoderHyperParameters.Prefix + "adapters.count"] = g.Int("adapters.count");
    }

    /// <inheritdoc/>
    public override WeightInfo MapWeight(WeightInfo w, GgufMetadataView g)
    {
        bool query = w.Name.EndsWith("attention.q.lora_b", StringComparison.Ordinal);
        if (!query && !w.Name.EndsWith("attention.k.lora_b", StringComparison.Ordinal))
        {
            return base.MapWeight(w, g);
        }

        // [S, N, R]: each slot's output rows are permuted per head like the base projection's.
        long slots = w.Shape[0].Fixed;
        long rows = w.Shape[1].Fixed;
        int heads = query ? g.Int("attention.head_count") : g.Int("attention.head_count_kv");
        var flat = w.As(w.Name, new Shape(slots * rows, w.Shape[2].Fixed));
        return DecoderGgufMapping.UnpermuteRows(flat, checked((int)(slots * heads))).As(w.Name, w.Shape);
    }

    /// <inheritdoc/>
    protected override IEnumerable<WeightInfo> FamilySyntheticWeights(GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        int vocabulary = checked((int)weights["token_embedding.weight"].Shape[0].Fixed);
        yield return AdapterTokenTable.Create(vocabulary, Ints(g, "adapters.token_ids_activate"), Ints(g, "adapters.token_ids_substitute"),
            g.Float("adapters.router_gain", 15f));
    }

    private static int[] Ints(GgufMetadataView g, string key)
        => g.Raw(key) is Array values ? values.Cast<object>().Select(Convert.ToInt32).ToArray() : throw new InvalidDataException($"graniteswitch: {key} is missing.");
}
