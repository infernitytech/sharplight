using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits the audio tower graph (<see cref="ChunkedAudioTowerGraph.AudioEncodeGraph"/>) of a causal streaming audio language model (Voxtral
/// Realtime, <see cref="AudioTowerParameters.Causal"/>): log-mel frames [F, mel bins] (F even, a symbol) through two causally padded <c>Conv1d</c>
/// layers (strides 1 and 2, GELU), pre-norm transformer layers with RMS norms, rotary positions, SwiGLU MLPs and attention that sees only the last
/// <see cref="AudioTowerParameters.SlidingWindow"/> rows (row spans), the output RMS norm, then every <see cref="AudioTowerParameters.Downsample"/>
/// consecutive rows concatenated and the two-layer projector (no biases) into the language model's width. Because the attention is causal, the output
/// of an earlier row never depends on later audio: a recording can be encoded in pieces with the same result as in one pass.
/// </summary>
internal static class CausalAudioTowerGraph
{
    /// <summary>The f32 kernel views of the convolutions (same as the Whisper tower's).</summary>
    /// <param name="p">The tower parameters.</param>
    /// <param name="weight">Looks up a weight.</param>
    public static IEnumerable<WeightInfo> DerivedWeights(AudioTowerParameters p, Func<string, WeightInfo?> weight) => WhisperAudioTowerGraph.DerivedWeights(p, weight);

    /// <summary>Builds the graph.</summary>
    /// <param name="p">The tower parameters.</param>
    /// <param name="weights">Looks up the canonical and derived weights.</param>
    public static ComputeGraph Build(AudioTowerParameters p, Func<string, WeightInfo?> weights)
    {
        const string P = AudioTowerParameters.WeightPrefix;
        var b = new GraphBuilder(ChunkedAudioTowerGraph.AudioEncodeGraph, weights);
        Value x;
        Value positions;
        Value spans;
        using (b.Group("audio.stem"))
        {
            x = b.Input(ChunkedAudioTowerGraph.FeaturesInput, InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.AudioFrameSymbol), p.MelBins));
            for (int i = 0; i < p.ConvCount; i++)
            {
                string bias = $"{P}conv.{i}.bias";
                Value[] inputs = b.HasWeight(bias) ? [x, b.Weight(WhisperAudioTowerGraph.KernelName(i)), b.Weight(bias)] : [x, b.Weight(WhisperAudioTowerGraph.KernelName(i))];
                long stride = i == 0 ? 1L : 2L;
                var attributes = Attributes.From(("stride", stride), ("pad_left", 3L - stride), ("pad_right", 0L));
                x = TowerBlocks.Activation(b, b.Emit1("Conv1d", inputs, attributes, $"audio.conv{i}"), "gelu");
            }

            positions = b.Input(ChunkedAudioTowerGraph.PositionsInput, InputRole.Positions, DataType.I32, new Shape(x.Shape[0], 1));
            spans = b.Input(ChunkedAudioTowerGraph.SpansInput, InputRole.RowSpans, DataType.I32, new Shape(x.Shape[0], 2));
        }

        x = Layers(b, x, p, positions, spans);
        x = b.RmsNorm(x, b.Weight(P + "output_norm.weight"), p.Eps, "audio.output_norm");
        x = b.Emit1("Reshape", [x], Attributes.From(("shape", new long[] { 0, (long)p.Hidden * p.Downsample }), ("outer_symbol", WellKnown.EncoderSymbol)));
        x = AudioTowerLayers.Linear(b, x, P + "projector.1", "audio.projector1");
        x = TowerBlocks.Activation(b, x, p.Activation);
        x = AudioTowerLayers.Linear(b, x, P + "projector.2", ChunkedAudioTowerGraph.EmbeddingsOutput);
        b.Output(ChunkedAudioTowerGraph.EmbeddingsOutput, x);
        return b.Build();
    }

    private static Value Layers(GraphBuilder b, Value x, AudioTowerParameters p, Value positions, Value spans)
    {
        const string P = AudioTowerParameters.WeightPrefix;
        int inner = p.Heads * p.HeadSize;
        float scale = 1f / MathF.Sqrt(p.HeadSize);
        var rope = Attributes.From(("dims", p.HeadSize), ("base", p.RopeBase), ("style", "neox"));
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"{P}layers.{il}");
            string l = $"{P}layers.{il}.";
            var h = b.RmsNorm(x, b.Weight(l + "input_norm.weight"), p.Eps);
            var q = b.Emit1("Rope", [b.Reshape(AudioTowerLayers.Linear(b, h, l + "attention.q"), 0, p.Heads, p.HeadSize), positions], rope);
            var k = b.Emit1("Rope", [b.Reshape(AudioTowerLayers.Linear(b, h, l + "attention.k"), 0, p.Heads, p.HeadSize), positions], rope);
            var v = b.Reshape(AudioTowerLayers.Linear(b, h, l + "attention.v"), 0, p.Heads, p.HeadSize);
            var attention = b.Emit1("Attention", [q, k, v, spans], Attributes.From(("scale", scale), ("causal", false)), $"{l}attn");
            x = b.Add(x, AudioTowerLayers.Linear(b, b.Reshape(attention, 0, inner), l + "attention.o"), $"{l}attn_residual");
            h = b.RmsNorm(x, b.Weight(l + "post_attention_norm.weight"), p.Eps);
            var gate = TowerBlocks.Activation(b, AudioTowerLayers.Linear(b, h, l + "mlp.gate"), "silu");
            var mlp = AudioTowerLayers.Linear(b, b.Mul(gate, AudioTowerLayers.Linear(b, h, l + "mlp.up")), l + "mlp.down");
            x = b.Add(x, mlp, $"{l}out");
        }

        return x;
    }
}
