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
/// [rows, H]. The convolutions are the generic time-major <c>Conv1d</c> op over the features transposed to [frames, C]; f16 and bf16 kernels
/// are widened to f32 when loaded.
/// </remarks>
internal sealed class AudioConvStem
{
    private readonly long[] _strides;
    private readonly string _activation;

    /// <summary>Reads the stem's hyper-parameters and adds its kernel views to <paramref name="weights"/>.</summary>
    /// <param name="src">The model source.</param>
    /// <param name="weights">Receives the f32 [O, C, K] kernels <c>encoder.conv.{i}.kernel</c>.</param>
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

            weights[p + "kernel"] = w.Type == DataType.F32 ? w.As(p + "kernel", w.Shape) : WeightConversions.ToF32(w, p + "kernel", w.Shape);
        }
    }

    /// <summary>
    /// Emits the stem over <paramref name="features"/> [C, F] and returns the rows [F / Π strides, H] with the positions added.
    /// </summary>
    /// <param name="b">The graph builder (the kernel views must be resolvable through it).</param>
    /// <param name="features">The audio features [channels, frames].</param>
    /// <param name="activation">Emits the activation of the given name.</param>
    public Value Build(GraphBuilder b, Value features, Func<Value, string, Value> activation)
    {
        Value x;
        using (b.Group("encoder.conv"))
        {
            x = b.Permute(features, 1, 0);
            for (int i = 0; i < _strides.Length; i++)
            {
                string p = $"encoder.conv.{i}.";
                var kernel = b.Weight(p + "kernel");
                long pad = kernel.Shape[2].Fixed / 2;
                Value[] inputs = b.HasWeight(p + "bias") ? [x, kernel, b.Weight(p + "bias")] : [x, kernel];
                var attributes = Attributes.From(("stride", _strides[i]), ("pad_left", pad), ("pad_right", pad));
                x = b.Emit1("Conv1d", inputs, attributes, $"conv{i}");
                x = activation(x, _activation);
            }
        }

        using (b.Group("input"))
        {
            return b.Add(x, b.Weight("encoder.position_embedding.weight"), "inp_embd");
        }
    }
}
