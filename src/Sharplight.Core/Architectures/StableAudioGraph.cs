using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits Stable Audio Open as graphs of generic ops over time-major signals: the timing conditioner (a learned Fourier embedding of the normalized seconds and a linear
/// layer per number), one evaluation of the diffusion transformer (a global row of the timing and the network's time prepended to the projected latents, pre-norm blocks
/// of self-attention with a partial rotary embedding, grouped cross-attention over the conditioning rows and a SwiGLU feed-forward layer, 1×1 convolutions around it)
/// and the autoencoder's decoder (<see cref="OobleckDecoderGraph"/>). The text encoder is <see cref="T5TextGraphBuilder"/>. Contract: <see cref="TimedAudioDiffusionGraphs"/>.
/// </summary>
internal static class StableAudioGraph
{
    /// <summary>The timing conditioner: the start and the total seconds to two rows of the conditioning width.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildTiming(StableAudioParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(TimedAudioDiffusionGraphs.TimingGraph, weights);
        var seconds = b.Input(TimedAudioDiffusionGraphs.SecondsInput, InputRole.Other, DataType.F32, new Shape(2));
        float range = p.MaxSeconds - p.MinSeconds;
        var normalized = b.Reshape(b.Scale(seconds, 1f / range, -p.MinSeconds / range), 2, 1);
        var start = Number(b, b.Slice(normalized, 0, 0, 1), "timing.start.");
        var total = Number(b, b.Slice(normalized, 0, 1, 1), "timing.total.");
        b.Output(TimedAudioDiffusionGraphs.TimingOutput, b.Concat(0, start, total));
        return b.Build();
    }

    /// <summary>One evaluation of the diffusion transformer: preconditioned latents to the v prediction.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="promptLength">The prompt rows of the conditioning (the timing rows follow them).</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDenoiser(StableAudioParameters p, int promptLength, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(TimedAudioDiffusionGraphs.DenoiseGraph, weights);
        var latent = b.Input(TimedAudioDiffusionGraphs.LatentInput, InputRole.Other, DataType.F32, new Shape(p.Frames, p.LatentChannels));
        var timestep = b.Input(TimedAudioDiffusionGraphs.TimestepInput, InputRole.Other, DataType.F32, new Shape(1));
        var condition = b.Input(TimedAudioDiffusionGraphs.ConditionInput, InputRole.Other, DataType.F32, new Shape(promptLength + 2, p.ConditionWidth));
        var global = b.Input(TimedAudioDiffusionGraphs.GlobalInput, InputRole.Other, DataType.F32, new Shape(1, 2 * p.ConditionWidth));
        var positions = b.Input(TimedAudioDiffusionGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(p.Frames + 1, 1));

        var context = Mlp2(b, condition, "dit.condition_embedding.");
        var globalRow = b.Add(Mlp2(b, global, "dit.global_embedding."), TimeEmbedding(b, timestep));
        var x = b.Add(latent, b.MatMul(latent, b.Weight("dit.input_mix.weight")));
        x = b.Concat(0, globalRow, b.MatMul(x, b.Weight("dit.proj_in.weight")));
        var rope = Attributes.From(("dims", (long)p.RopeDims), ("base", p.RopeBase), ("style", "neox"));
        var attention = Attributes.From(("scale", 1f / MathF.Sqrt(p.HeadDim)), ("causal", false));
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"dit.layers.{il}");
            string n = $"dit.layers.{il}.";
            var h = Norm(b, x, n + "attention_norm", p.NormEps);
            var q = b.Emit1("Rope", [Heads(b, h, n + "attention.q.weight", p.Heads, p.HeadDim), positions], rope);
            var k = b.Emit1("Rope", [Heads(b, h, n + "attention.k.weight", p.Heads, p.HeadDim), positions], rope);
            var v = Heads(b, h, n + "attention.v.weight", p.Heads, p.HeadDim);
            var attended = b.Emit1("Attention", [q, k, v], attention);
            x = b.Add(x, b.MatMul(b.Reshape(attended, 0, p.Heads * p.HeadDim), b.Weight(n + "attention.o.weight")));

            h = Norm(b, x, n + "cross_attention_norm", p.NormEps);
            q = Heads(b, h, n + "cross_attention.q.weight", p.Heads, p.HeadDim);
            k = Heads(b, context, n + "cross_attention.k.weight", p.KvHeads, p.HeadDim);
            v = Heads(b, context, n + "cross_attention.v.weight", p.KvHeads, p.HeadDim);
            attended = b.Emit1("Attention", [q, k, v], attention);
            x = b.Add(x, b.MatMul(b.Reshape(attended, 0, p.Heads * p.HeadDim), b.Weight(n + "cross_attention.o.weight")));

            h = Norm(b, x, n + "mlp_norm", p.NormEps);
            var up = b.Linear(h, n + "mlp.up.weight", n + "mlp.up.bias");
            long inner = b.Weight(n + "mlp.down.weight").Shape[1].Fixed;
            var gated = b.Mul(b.Slice(up, 1, 0, inner), b.Silu(b.Slice(up, 1, inner, inner)));
            x = b.Add(x, b.Linear(gated, n + "mlp.down.weight", n + "mlp.down.bias"));
        }

        var output = b.Slice(b.MatMul(x, b.Weight("dit.proj_out.weight")), 0, 1, p.Frames);
        output = b.Add(output, b.MatMul(output, b.Weight("dit.output_mix.weight")));
        b.Output(TimedAudioDiffusionGraphs.PredictionOutput, output);
        return b.Build();
    }

    /// <summary>The autoencoder's decoder: latents <c>[T, C]</c> to audio <c>[T · samples per frame, channels]</c>.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDecoder(StableAudioParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioGenerationGraphs.DecodeGraph, weights);
        var x = b.Input(AudioGenerationGraphs.LatentInput, InputRole.Other, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), p.LatentChannels));
        b.Output(AudioGenerationGraphs.AudioOutput, OobleckDecoderGraph.Build(b, x, "vae.decoder.", p.VaeUpsampling));
        return b.Build();
    }

    // One number conditioner: [n, sin(2π n w), cos(2π n w)] through a linear layer, n the normalized number [1, 1].
    private static Value Number(GraphBuilder b, Value normalized, string n)
    {
        var frequencies = b.Weight(n + "frequencies");
        long half = frequencies.Shape[0].Fixed;
        var angles = b.Mul(b.Scale(normalized, 2 * MathF.PI), b.Reshape(frequencies, 1, half));
        var features = b.Concat(1, normalized, b.Unary("Sin", angles), b.Unary("Cos", angles));
        return b.Linear(features, n + "projection.weight", n + "projection.bias");
    }

    // The network's time as Gaussian Fourier features [cos(2π t w), sin(2π t w)] through two linear layers with SiLU: [1, hidden].
    private static Value TimeEmbedding(GraphBuilder b, Value timestep)
    {
        var frequencies = b.Weight("dit.time_embedding.frequencies");
        long half = frequencies.Shape[0].Fixed;
        var angles = b.Mul(b.Reshape(b.Scale(timestep, 2 * MathF.PI), 1, 1), b.Reshape(frequencies, 1, half));
        var features = b.Concat(1, b.Unary("Cos", angles), b.Unary("Sin", angles));
        var hidden = b.Linear(features, "dit.time_embedding.linear_1.weight", "dit.time_embedding.linear_1.bias");
        return b.Linear(b.Silu(hidden), "dit.time_embedding.linear_2.weight", "dit.time_embedding.linear_2.bias");
    }

    // Two bias-free linear layers with SiLU between (the conditioning projections).
    private static Value Mlp2(GraphBuilder b, Value x, string n)
        => b.MatMul(b.Silu(b.MatMul(x, b.Weight(n + "linear_1.weight"))), b.Weight(n + "linear_2.weight"));

    private static Value Norm(GraphBuilder b, Value x, string n, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(n + ".weight"), b.Weight(n + ".bias")], Attributes.From(("eps", eps)));

    private static Value Heads(GraphBuilder b, Value x, string weight, int heads, int headDim) => b.Reshape(b.MatMul(x, b.Weight(weight)), 0, heads, headDim);
}
