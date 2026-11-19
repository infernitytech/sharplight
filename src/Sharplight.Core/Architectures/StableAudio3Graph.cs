using System.Runtime.InteropServices;
using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits Stable Audio 3's seconds conditioner and one evaluation of its diffusion transformer as graphs of generic ops over time-major signals. The transformer: the
/// latents through a residual 1×1 convolution and a linear layer, the learned memory rows appended, blocks of adaLN-modulated self-attention (RMS-normed queries and
/// keys, a partial rotary embedding, padding frames' values zeroed, the output gated by <c>sigmoid(1 − g)</c>), cross-attention over the conditioning rows, the local
/// additive conditioning and an adaLN-modulated SwiGLU feed-forward layer; a linear layer and a residual 1×1 convolution at the end. The modulation of every block is its
/// learned table plus an MLP of the global row (the seconds row through two layers plus the time embedding). Contract: <see cref="TimedAudioDiffusionGraphs"/>.
/// </summary>
internal static class StableAudio3Graph
{
    /// <summary>The seconds conditioner: the total seconds f32 <c>[1]</c> to one row <c>[1, width]</c> (exponential Fourier features of the clamped, normalized seconds
    /// through a linear layer).</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildSeconds(StableAudio3Parameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(TimedAudioDiffusionGraphs.TimingGraph, weights);
        var seconds = b.Input(TimedAudioDiffusionGraphs.SecondsInput, InputRole.Other, DataType.F32, new Shape(1));
        var clamped = b.Binary("Max", b.Binary("Min", seconds, Scalar(b, "seconds_max", p.MaxSeconds)), Scalar(b, "seconds_min", p.MinSeconds));
        var normalized = b.Binary("Div", b.Sub(clamped, Scalar(b, "seconds_min_offset", p.MinSeconds)), Scalar(b, "seconds_range", p.MaxSeconds - p.MinSeconds));
        var features = FourierFeatures(b, normalized, p.TimeFeatures, "seconds_frequencies");
        b.Output(TimedAudioDiffusionGraphs.TimingOutput, b.Linear(features, "seconds.projection.weight", "seconds.projection.bias"));
        return b.Build();
    }

    /// <summary>One evaluation of the diffusion transformer: the latents at time t to the velocity.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDenoiser(StableAudio3Parameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(TimedAudioDiffusionGraphs.DenoiseGraph, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        var rows = frames + Dim.Of(p.MemoryTokens);
        int D = p.Hidden;
        var latent = b.Input(TimedAudioDiffusionGraphs.LatentInput, InputRole.Other, DataType.F32, new Shape(frames, p.LatentChannels));
        var timestep = b.Input(TimedAudioDiffusionGraphs.TimestepInput, InputRole.Other, DataType.F32, new Shape(1));
        var condition = b.Input(TimedAudioDiffusionGraphs.ConditionInput, InputRole.Other, DataType.F32, new Shape(p.PromptLength + 1, p.ConditionWidth));
        var global = b.Input(TimedAudioDiffusionGraphs.GlobalInput, InputRole.Other, DataType.F32, new Shape(1, p.ConditionWidth));
        var local = b.Input(TimedAudioDiffusionGraphs.LocalInput, InputRole.Other, DataType.F32, new Shape(frames, p.LocalChannels));
        var positions = b.Input(TimedAudioDiffusionGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(rows, 1));
        var valid = b.Input(TimedAudioDiffusionGraphs.ValidInput, InputRole.Other, DataType.F32, new Shape(rows));

        Value context, modulation, x, validValues, memoryZeros;
        using (b.Group("dit.input"))
        {
            context = Mlp2(b, condition, "dit.condition_embedding.");
            var time = FourierFeatures(b, timestep, p.TimeFeatures, "time_frequencies");
            var timeEmbedding = Mlp2(b, time, "dit.time_embedding.");
            var globalRow = b.Add(Mlp2(b, global, "dit.global_embedding."), timeEmbedding);
            modulation = Mlp2(b, globalRow, "dit.modulation_embedding.");
            x = b.Add(latent, b.MatMul(latent, b.Weight("dit.input_mix.weight")));
            x = b.MatMul(x, b.Weight("dit.proj_in.weight"));
            // The memory rows follow the frames (their positions, given by the caller, are 0 … M − 1): the frames stay a prefix of the sequence.
            x = b.Concat(0, x, b.Weight("dit.memory_tokens"));
            validValues = b.Reshape(valid, 0, 1, 1);
            memoryZeros = b.Constant("memory_zeros", DataType.F32, new Shape(p.MemoryTokens, D), new byte[p.MemoryTokens * D * sizeof(float)]);
        }

        var rope = Attributes.From(("dims", (long)p.RopeDims), ("base", StableAudio3Parameters.RopeBase), ("style", "neox"));
        var attention = Attributes.From(("scale", 1f / MathF.Sqrt(p.HeadDim)), ("causal", false));
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"dit.layers.{il}");
            string n = $"dit.layers.{il}.";
            var mod = b.Add(b.Reshape(b.Weight(n + "modulation"), 1, 6 * D), modulation);
            Value Chunk(int i) => b.Slice(mod, 1, i * D, D);

            // Self-attention with adaLN.
            var h = Modulate(b, b.RmsNorm(x, b.Weight(n + "attention_norm.weight"), StableAudio3Parameters.NormEps), Chunk(0), Chunk(1));
            var qkv = b.MatMul(h, b.Weight(n + "attention.qkv.weight"));
            Value Part(int i) => b.Reshape(b.Slice(qkv, 1, i * D, D), 0, p.Heads, p.HeadDim);
            var v = b.Mul(Part(2), validValues);
            var attended = SelfAttention(b, n, Part(0), Part(1), v, positions, rope, attention);
            if (p.Differential)
            {
                attended = b.Sub(attended, SelfAttention(b, n, Part(3), Part(4), v, positions, rope, attention));
            }

            var o = b.MatMul(b.Reshape(attended, 0, D), b.Weight(n + "attention.o.weight"));
            x = b.Add(x, b.Mul(o, Gate(b, Chunk(2))));

            // Cross-attention over the conditioning rows.
            h = b.RmsNorm(x, b.Weight(n + "cross_attention_norm.weight"), StableAudio3Parameters.NormEps);
            var q = b.MatMul(h, b.Weight(n + "cross_attention.q.weight"));
            var kv = b.MatMul(context, b.Weight(n + "cross_attention.kv.weight"));
            Value QPart(int i) => QkNorm(b, b.Reshape(b.Slice(q, 1, i * D, D), 0, p.Heads, p.HeadDim), n + "cross_attention.q_norm.weight");
            Value KvPart(int i) => b.Reshape(b.Slice(kv, 1, i * D, D), 0, p.Heads, p.HeadDim);
            int kvValue = p.Differential ? 2 : 1;
            var cv = KvPart(kvValue);
            attended = b.Emit1("Attention", [QPart(0), QkNorm(b, KvPart(0), n + "cross_attention.k_norm.weight"), cv], attention);
            if (p.Differential)
            {
                attended = b.Sub(attended, b.Emit1("Attention", [QPart(1), QkNorm(b, KvPart(1), n + "cross_attention.k_norm.weight"), cv], attention));
            }

            x = b.Add(x, b.MatMul(b.Reshape(attended, 0, D), b.Weight(n + "cross_attention.o.weight")));

            // Local additive conditioning on the frames (not the memory rows).
            if (p.LocalChannels > 0)
            {
                x = b.Add(x, b.Concat(0, Mlp2(b, local, n + "local_embedding."), memoryZeros));
            }

            // Feed-forward layer with adaLN.
            h = Modulate(b, b.RmsNorm(x, b.Weight(n + "mlp_norm.weight"), StableAudio3Parameters.NormEps), Chunk(3), Chunk(4));
            var up = b.Linear(h, n + "mlp.up.weight", n + "mlp.up.bias");
            long inner = b.Weight(n + "mlp.down.weight").Shape[1].Fixed;
            var gated = b.Mul(b.Slice(up, 1, 0, inner), b.Silu(b.Slice(up, 1, inner, inner)));
            x = b.Add(x, b.Mul(b.Linear(gated, n + "mlp.down.weight", n + "mlp.down.bias"), Gate(b, Chunk(5))));
        }

        using (b.Group("dit.output"))
        {
            var output = b.MatMul(x, b.Weight("dit.proj_out.weight"));
            output = b.Add(output, b.MatMul(output, b.Weight("dit.output_mix.weight")));
            b.Output(TimedAudioDiffusionGraphs.PredictionOutput, output);
        }

        return b.Build();
    }

    /// <summary>
    /// Exponential Fourier features of a value <c>[1]</c>: <c>[cos(2π v f), sin(2π v f)]</c> over <c>width / 2</c> frequencies spaced exponentially from 0.5 to 10000,
    /// as the reference computes them in f32 (<c>torch.linspace</c>'s ramp, the angle as <c>((v · f) · 2) · π</c>).
    /// </summary>
    private static Value FourierFeatures(GraphBuilder b, Value value, int width, string name)
    {
        int half = width / 2;
        var frequencies = new float[half];
        float step = 1f / (half - 1);
        float logMin = MathF.Log(StableAudio3Parameters.MinFrequency);
        float logRange = (float)(Math.Log(StableAudio3Parameters.MaxFrequency) - Math.Log(StableAudio3Parameters.MinFrequency));
        for (int i = 0; i < half; i++)
        {
            float ramp = i < half / 2 ? step * i : 1f - (step * (half - i - 1));
            frequencies[i] = (float)Math.Exp((ramp * logRange) + logMin);
        }

        var f = b.Constant(name, DataType.F32, new Shape(1, half), MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var angles = b.Scale(b.Scale(b.Mul(b.Reshape(value, 1, 1), f), 2f), MathF.PI);
        return b.Concat(1, b.Unary("Cos", angles), b.Unary("Sin", angles));
    }

    private static Value SelfAttention(GraphBuilder b, string n, Value q, Value k, Value v, Value positions, Attributes rope, Attributes attention)
    {
        q = b.Emit1("Rope", [QkNorm(b, q, n + "attention.q_norm.weight"), positions], rope);
        k = b.Emit1("Rope", [QkNorm(b, k, n + "attention.k_norm.weight"), positions], rope);
        return b.Emit1("Attention", [q, k, v], attention);
    }

    private static Value QkNorm(GraphBuilder b, Value x, string weight) => b.RmsNorm(x, b.Weight(weight), StableAudio3Parameters.QkNormEps);

    // x · (1 + scale) + shift.
    private static Value Modulate(GraphBuilder b, Value x, Value scale, Value shift) => b.Add(b.Mul(x, b.Scale(scale, 1f, 1f)), shift);

    // sigmoid(1 − g).
    private static Value Gate(GraphBuilder b, Value g) => b.Sigmoid(b.Scale(g, -1f, 1f));

    // Two linear layers with SiLU between (their biases when the model has them).
    private static Value Mlp2(GraphBuilder b, Value x, string n)
        => b.Linear(b.Silu(b.Linear(x, n + "linear_1.weight", n + "linear_1.bias")), n + "linear_2.weight", n + "linear_2.bias");

    private static Value Scalar(GraphBuilder b, string name, float value)
        => b.Constant(name, DataType.F32, new Shape(1), MemoryMarshal.AsBytes(new[] { value }.AsSpan()).ToArray());
}
