using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The convolution stem of speech encoders over frame-level audio features: 1-D convolutions along time (padding K / 2, each followed by
/// an activation), then the result as rows [frames / Π strides, H] plus learned positions. Whisper: two kernel-3 convolutions with strides
/// 1 and 2 and GELU, 3000 log-mel frames to 1500 rows.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>{arch}.</c>): <c>encoder.conv.strides</c> (one per convolution), optional <c>encoder.conv.activation</c> (default
/// "gelu"). Weights: <c>encoder.conv.{i}.weight</c> [O, C, K] with optional <c>.bias</c> [O]; <c>encoder.position_embedding.weight</c>
/// [rows, H]. The 1-D kernels run as [O, C, 1, K] kernels of the 2-D <c>Conv</c> op over features viewed as [1, C, 1, frames]; f16 and
/// bf16 kernels are widened to f32 when loaded.
/// </remarks>
internal sealed class AudioConvStem
{
    private readonly long[] _strides;
    private readonly string _activation;

    /// <summary>Reads the stem's hyper-parameters and adds its kernel views to <paramref name="weights"/>.</summary>
    /// <param name="src">The model source.</param>
    /// <param name="weights">Receives the [O, C, 1, K] kernel views <c>encoder.conv.{i}.kernel</c>.</param>
    public AudioConvStem(ModelSource src, IDictionary<string, WeightInfo> weights)
    {
        _strides = src.Arch<long[]>("encoder.conv.strides");
        _activation = src.Arch("encoder.conv.activation", "gelu");
        for (int i = 0; i < _strides.Length; i++)
        {
            string p = $"encoder.conv.{i}.";
            var w = src.Weight(p + "weight") ?? throw new NotSupportedException($"{src.Architecture}: missing '{p}weight'.");
            if (w.Shape.Rank != 3)
            {
                throw new NotSupportedException($"{src.Architecture}: '{p}weight' must be [out, in, kernel], got {w.Shape}.");
            }

            var kernelShape = new Shape(w.Shape[0], w.Shape[1], 1, w.Shape[2]);
            weights[p + "kernel"] = w.Type == DataType.F32 ? w.As(p + "kernel", kernelShape) : WeightConversions.ToF32(w, p + "kernel", kernelShape);
        }
    }

    /// <summary>
    /// Emits the stem over <paramref name="features"/> [C, F] and returns the rows [F / Π strides, H] with the positions added.
    /// </summary>
    /// <param name="b">The graph builder (the kernel views must be resolvable through it).</param>
    /// <param name="features">The audio features [channels, frames].</param>
    /// <param name="hidden">The model width H.</param>
    /// <param name="activation">Emits the activation of the given name.</param>
    public Value Build(GraphBuilder b, Value features, int hidden, Func<Value, string, Value> activation)
    {
        Value x;
        using (b.Group("encoder.conv"))
        {
            x = b.Reshape(features, 1, features.Shape[0].Fixed, 1, features.Shape[1].Fixed);
            for (int i = 0; i < _strides.Length; i++)
            {
                string p = $"encoder.conv.{i}.";
                var kernel = b.Weight(p + "kernel");
                long pad = kernel.Shape[3].Fixed / 2;
                Value[] inputs = b.HasWeight(p + "bias") ? [x, kernel, b.Weight(p + "bias")] : [x, kernel];
                x = b.Emit1("Conv", inputs, Attributes.From(("strides", new long[] { 1, _strides[i] }), ("pads", new long[] { 0, pad, 0, pad })), $"conv{i}");
                x = activation(x, _activation);
            }
        }

        using (b.Group("input"))
        {
            var position = b.Weight("encoder.position_embedding.weight");
            long rows = position.Shape[0].Fixed;
            var rowMajor = b.Permute(b.Reshape(x, hidden, rows), 1, 0);
            return b.Add(rowMajor, position, "inp_embd");
        }
    }
}
