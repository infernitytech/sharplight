using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face <c>granite_switch</c> (GraniteSwitchForCausalLM, remote code): a dense Granite decoder (muP multipliers) whose fused
/// <c>qkv_proj</c>, <c>o_proj</c>, fused gate/up <c>shared_mlp.input_linear</c> and <c>shared_mlp.output_linear</c> are LoRA-wrapped:
/// <c>base_layer.weight</c> plus adapter-stacked <c>lora_A</c> [adapters, 1, rank, in] / <c>lora_B</c> [adapters, 1, out, rank] (one pair per
/// slice of a fused projection, <c>lora_A_slices.i</c> / <c>lora_B_slices.i</c>). They become the canonical projections with their
/// <c>lora_a</c> / <c>lora_b</c> pairs, an all-zero slot 0 (the base model) in front. The adapters' activation tokens, substitute tokens and
/// router gain (<c>control_token_gain</c>) become the adapter token table; the weightless switch layers counted in <c>num_hidden_layers</c> are
/// left out, as are the switch's own buffers (the router is rebuilt from the token table, as llama.cpp does).
/// </summary>
internal sealed partial class GraniteSwitchHfMapping : HfDecoderFamilyMapping
{
    private static readonly string[] s_attentionSlices = ["attention.q", "attention.k", "attention.v"];
    private static readonly string[] s_mlpSlices = ["mlp.gate", "mlp.up"];

    /// <inheritdoc/>
    public override IReadOnlyList<string> ModelTypes { get; } = ["granite_switch"];

    /// <inheritdoc/>
    protected override int FeedForwardSize(HfConfig c) => c.GetInt("shared_intermediate_size") ?? c.Int("intermediate_size");

    /// <inheritdoc/>
    protected override bool IsIgnored(string name) => base.IsIgnored(name) || name.StartsWith("model.switch.", StringComparison.Ordinal);

    /// <inheritdoc/>
    protected override void AddHyperParameters(Dictionary<string, object> md, HfConfig c)
    {
        const string prefix = "decoder.";
        GraniteHfMapping.AddMupScales(md, c);
        md[prefix + "layer_count"] = WeightLayerCount(c);
        md[prefix + "adapters.count"] = c.Int("num_adapters");
        if (c.GetInt("projection_head_dim") is { } headDim)
        {
            md[prefix + "attention.head_dim"] = headDim;
            md[prefix + "rope.dim"] = headDim;
        }
    }

    /// <inheritdoc/>
    protected override bool TryMapSpecialLayerTensor(HfConfig c, string layerPrefix, string suffix, WeightInfo w, HfWeightCollector collector)
    {
        int headDim = c.GetInt("projection_head_dim") ?? HeadDimOf(c);
        int qRows = c.Int("num_attention_heads") * headDim;
        int kvRows = (c.GetInt("num_key_value_heads") ?? c.Int("num_attention_heads")) * headDim;
        int ffn = FeedForwardSize(c);
        switch (suffix)
        {
            case "self_attn.qkv_proj.base_layer.weight":
                collector.Add(w.RowSlice(layerPrefix + "attention.q.weight", 0, qRows));
                collector.Add(w.RowSlice(layerPrefix + "attention.k.weight", qRows, kvRows));
                collector.Add(w.RowSlice(layerPrefix + "attention.v.weight", qRows + kvRows, kvRows));
                return true;
            case "self_attn.o_proj.base_layer.weight":
                collector.Add(w.As(layerPrefix + "attention.o.weight"));
                return true;
            case "self_attn.o_proj.lora_A" or "self_attn.o_proj.lora_B":
                collector.Add(WithZeroSlot(w, layerPrefix + "attention.o." + PairName(suffix)));
                return true;
            case "shared_mlp.input_linear.base_layer.weight":
                collector.Add(w.RowSlice(layerPrefix + "mlp.gate.weight", 0, ffn));
                collector.Add(w.RowSlice(layerPrefix + "mlp.up.weight", ffn, ffn));
                return true;
            case "shared_mlp.output_linear.base_layer.weight":
                collector.Add(w.As(layerPrefix + "mlp.down.weight"));
                return true;
            case "shared_mlp.output_linear.lora_A" or "shared_mlp.output_linear.lora_B":
                collector.Add(WithZeroSlot(w, layerPrefix + "mlp.down." + PairName(suffix)));
                return true;
        }

        var slice = SliceName().Match(suffix);
        if (!slice.Success)
        {
            return false;
        }

        var targets = slice.Groups[1].Value == "self_attn.qkv_proj" ? s_attentionSlices : s_mlpSlices;
        int index = int.Parse(slice.Groups[3].Value);
        if (index >= targets.Length)
        {
            return false;
        }

        collector.Add(WithZeroSlot(w, $"{layerPrefix}{targets[index]}.lora_{(slice.Groups[2].Value == "A" ? "a" : "b")}"));
        return true;
    }

    /// <inheritdoc/>
    protected override void FinishWeights(HfConfig c, HfWeightCollector collector)
    {
        base.FinishWeights(c, collector);
        var activation = IntArray(c, "adapter_token_ids") ?? throw new InvalidDataException("granite_switch: adapter_token_ids is missing.");
        var substitute = IntArray(c, "adapter_substitute_token_ids") ?? throw new InvalidDataException("granite_switch: adapter_substitute_token_ids is missing.");
        collector.Add(AdapterTokenTable.Create(c.Int("vocab_size"), activation, substitute, (float)(c.GetDouble("control_token_gain") ?? 15.0)));
    }

    private static string PairName(string suffix) => suffix.EndsWith("lora_A", StringComparison.Ordinal) ? "lora_a" : "lora_b";

    // [adapters, 1, rows, cols] → [adapters + 1, rows, cols] with the all-zero slot 0 (the base model) in front.
    private static WeightInfo WithZeroSlot(WeightInfo w, string name)
    {
        long adapters = w.Shape[0].Fixed;
        long rows = w.Shape[w.Shape.Rank - 2].Fixed;
        long cols = w.Shape[w.Shape.Rank - 1].Fixed;
        long slotBytes = w.Type.ByteSize(rows * cols);
        return WeightInfo.Derived(name, w.Type, new Shape(adapters + 1, rows, cols), () =>
        {
            var stacked = new byte[checked((int)((adapters + 1) * slotBytes))];
            w.ReadAll().CopyTo(stacked, slotBytes);
            return stacked;
        });
    }

    // The checkpoint's layers with weights (num_hidden_layers also counts the weightless switch layers).
    private static int WeightLayerCount(HfConfig c)
    {
        using var checkpoint = SafetensorsCheckpoint.Open(c.Directory);
        return checkpoint.Tensors.Keys.Select(name => LayerIndex().Match(name)).Where(m => m.Success).Select(m => int.Parse(m.Groups[1].Value) + 1)
            .DefaultIfEmpty(c.Int("num_hidden_layers")).Max();
    }

    [GeneratedRegex(@"^(self_attn\.qkv_proj|shared_mlp\.input_linear)\.lora_([AB])_slices\.(\d+)$")]
    private static partial Regex SliceName();

    [GeneratedRegex(@"^(?:model\.)?layers\.(\d+)\.")]
    private static partial Regex LayerIndex();
}
