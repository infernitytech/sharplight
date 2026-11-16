using Sharplight.Graph;

namespace Sharplight.Architectures;

/// <summary>The pre-norm transformer layers of audio towers (<see cref="AudioTowerParameters"/>), shared by the chunked and the Whisper-style tower.</summary>
internal static class AudioTowerLayers
{
    /// <summary>
    /// The layers <c>audio.layers.{i}.</c> over <paramref name="x"/> [rows, hidden]: LayerNorm, multi-head attention (biases where the checkpoint
    /// has them; within the row spans when given), residual, LayerNorm, MLP, residual.
    /// </summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="x">The rows.</param>
    /// <param name="p">The tower parameters.</param>
    /// <param name="spans">The attention span (first, count) of every row, or <see langword="null"/> to attend over all rows.</param>
    public static Value Build(GraphBuilder b, Value x, AudioTowerParameters p, Value? spans)
    {
        const string P = AudioTowerParameters.WeightPrefix;
        int headSize = p.Hidden / p.Heads;
        float scale = 1f / MathF.Sqrt(headSize);
        long rows = x.Shape[0].IsSymbolic ? 0 : x.Shape[0].Fixed;   // Reshape keeps a 0 dimension only when it is symbolic
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"{P}layers.{il}");
            string l = $"{P}layers.{il}.";
            var h = TowerBlocks.LayerNorm(b, x, l + "input_norm", p.Eps);
            var q = b.Reshape(Linear(b, h, l + "attention.q"), rows, p.Heads, headSize);
            var k = b.Reshape(Linear(b, h, l + "attention.k"), rows, p.Heads, headSize);
            var v = b.Reshape(Linear(b, h, l + "attention.v"), rows, p.Heads, headSize);
            Value[] inputs = spans is null ? [q, k, v] : [q, k, v, spans];
            var attention = b.Emit1("Attention", inputs, Attributes.From(("scale", scale), ("causal", false)), $"{l}attn");
            x = b.Add(x, Linear(b, b.Reshape(attention, rows, p.Hidden), l + "attention.o"), $"{l}attn_residual");
            h = TowerBlocks.LayerNorm(b, x, l + "post_attention_norm", p.Eps);
            var mlp = Linear(b, TowerBlocks.Activation(b, Linear(b, h, l + "mlp.up"), p.Activation), l + "mlp.down");
            x = b.Add(x, mlp, $"{l}out");
        }

        return x;
    }

    /// <summary>A linear layer <c>module.weight</c> with <c>module.bias</c> when the checkpoint has one.</summary>
    /// <param name="b">The graph builder.</param>
    /// <param name="x">The input rows.</param>
    /// <param name="module">The module name.</param>
    /// <param name="name">The output name, or <see langword="null"/>.</param>
    public static Value Linear(GraphBuilder b, Value x, string module, string? name = null)
        => b.Linear(x, module + ".weight", b.HasWeight(module + ".bias") ? module + ".bias" : null, name);
}
