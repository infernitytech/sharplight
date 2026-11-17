using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// XTTS v2's style encoder (<c>GPT.get_style_emb</c>) for one chunk of a recording: the log-mel divided by the model's mel statistics, the
/// conditioning encoder (a pointwise projection, then attention blocks that group-normalize, attend with the heads split before q, k and v, and add
/// the projection to the normalized input) and the Perceiver resampler (32 learned latents attending to themselves and the frames, GEGLU feed
/// forwards, an RMS norm scaled by √width). The host averages the latents of the chunks (<see cref="XttsVoiceCloning"/>).
/// </summary>
internal static class XttsConditioningGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "xtts_conditioning";

    /// <summary>The log-mel frames of a chunk [frames, 80] (frames symbolic: <see cref="WellKnown.TokenSymbol"/>).</summary>
    public const string MelInput = "mel";

    /// <summary>The chunk's latents [32, width].</summary>
    public const string LatentsOutput = "latents";

    private const string E = "gpt.conditioning_encoder.";
    private const string R = "gpt.conditioning_perceiver.";

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="width">The model width.</param>
    /// <param name="encoderBlocks">The conditioning encoder's attention blocks.</param>
    /// <param name="encoderHeads">Their heads.</param>
    /// <param name="perceiverLayers">The Perceiver layers.</param>
    /// <param name="perceiverHeads">Their heads.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, int width, int encoderBlocks, int encoderHeads, int perceiverLayers, int perceiverHeads)
    {
        var b = new GraphBuilder(GraphName, weights);
        var mel = b.Input(MelInput, InputRole.AudioFeatures, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), 80));
        var x = Pointwise(b, b.Binary("Div", mel, b.Weight("gpt.mel_stats")), E + "init");
        int headSize = width / encoderHeads;
        for (int i = 0; i < encoderBlocks; i++)
        {
            string n = $"{E}attn.{i}.";
            var normed = GroupNorm(b, x, n + "norm", 32, width);
            var qkv = b.Reshape(Pointwise(b, normed, n + "qkv"), 0, encoderHeads, 3 * headSize);
            Value Part(int index) => b.Slice(qkv, 2, index * headSize, headSize);
            var attention = b.Emit1("Attention", [Part(0), Part(1), Part(2)], Attributes.From(("scale", 1f / MathF.Sqrt(headSize)), ("causal", false)));
            x = b.Add(normed, Pointwise(b, b.Reshape(attention, 0, width), n + "proj_out"));
        }

        var latents = b.Weight(R + "latents");
        long queries = latents.Shape[0].Fixed;
        int inner = (int)weights(R + "layers.0.0.to_q.weight")!.Shape[0].Fixed;
        int perceiverHeadSize = inner / perceiverHeads;
        for (int l = 0; l < perceiverLayers; l++)
        {
            string n = $"{R}layers.{l}.";
            var context = b.Concat(0, latents, x);
            var q = b.Reshape(b.Linear(latents, n + "0.to_q.weight"), queries, perceiverHeads, perceiverHeadSize);
            var kv = b.Linear(context, n + "0.to_kv.weight");
            var k = b.Reshape(b.Slice(kv, 1, 0, inner), 0, perceiverHeads, perceiverHeadSize);
            var v = b.Reshape(b.Slice(kv, 1, inner, inner), 0, perceiverHeads, perceiverHeadSize);
            // The latents reach 1e10 before the final norm; the split-f16 flash kernels scale rows into f16's range before splitting.
            var attention = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(perceiverHeadSize)), ("causal", false)));
            latents = b.Add(latents, b.Linear(b.Reshape(attention, queries, inner), n + "0.to_out.weight"));
            var hidden = b.Linear(latents, n + "1.0.weight", n + "1.0.bias");
            long half = hidden.Shape[1].Fixed / 2;
            var gated = b.Mul(b.Slice(hidden, 1, 0, half), b.Unary("Gelu", b.Slice(hidden, 1, half, half)));
            latents = b.Add(latents, b.Linear(gated, n + "1.2.weight", n + "1.2.bias"));
        }

        // F.normalize(x) · √width · gamma is the RMS norm of x scaled by gamma.
        var normalized = b.RmsNorm(latents, b.Weight(R + "norm.gamma"), 1e-24f);
        b.Output(LatentsOutput, normalized);
        return b.Build();
    }

    // A kernel-1 Conv1d (weight [O, C, 1], bias [O]) over time-major rows.
    private static Value Pointwise(GraphBuilder b, Value x, string module)
        => b.Emit1("Conv1d", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.Empty);

    // GroupNorm over time-major [frames, C]: each group of C / groups channels normalized over its channels and all frames, then scaled per channel.
    private static Value GroupNorm(GraphBuilder b, Value x, string module, int groups, int channels)
    {
        var grouped = b.Reshape(x, 0, groups, channels / groups);
        var axes = new long[] { 0, 2 };
        var mean = b.Emit1("Reduce", [grouped], Attributes.From(("mode", "mean"), ("axes", axes), ("keepdims", true)));
        var centered = b.Sub(grouped, mean);
        var variance = b.Emit1("Reduce", [b.Mul(centered, centered)], Attributes.From(("mode", "mean"), ("axes", axes), ("keepdims", true)));
        var normed = b.Reshape(b.Mul(centered, b.Unary("Rsqrt", b.Scale(variance, 1f, 1e-5f))), 0, channels);
        return b.Add(b.Mul(normed, b.Weight(module + ".weight")), b.Weight(module + ".bias"));
    }
}
