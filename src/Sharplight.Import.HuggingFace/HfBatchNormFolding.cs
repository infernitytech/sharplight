using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>Folds the inference-time batch normalization that follows a convolution into the convolution's weight and bias.</summary>
internal static class HfBatchNormFolding
{
    /// <summary>
    /// <c>w' = w · γ / √(σ² + ε)</c> per output channel and <c>b' = β − μ · γ / √(σ² + ε)</c>, as two lazily computed F32 weights.
    /// </summary>
    /// <param name="conv">The convolution weight [O, I, KH, KW].</param>
    /// <param name="gamma">The normalization scale [O].</param>
    /// <param name="beta">The normalization shift [O].</param>
    /// <param name="mean">The running mean [O].</param>
    /// <param name="variance">The running variance [O].</param>
    /// <param name="epsilon">The variance epsilon.</param>
    /// <param name="name">The canonical name of the folded convolution (without <c>.weight</c> / <c>.bias</c>).</param>
    public static (WeightInfo Weight, WeightInfo Bias) Fold(WeightInfo conv, WeightInfo gamma, WeightInfo beta, WeightInfo mean, WeightInfo variance, float epsilon, string name)
    {
        long outputs = conv.Shape[0].Fixed;
        float[] Scale()
        {
            var g = HfTransforms.DecodeF32(gamma);
            var v = HfTransforms.DecodeF32(variance);
            return [.. g.Select((value, i) => value / MathF.Sqrt(v[i] + epsilon))];
        }

        var weight = WeightInfo.Derived(name + ".weight", DataType.F32, conv.Shape, () =>
        {
            var w = HfTransforms.DecodeF32(conv);
            var scale = Scale();
            long per = w.Length / outputs;
            for (long o = 0; o < outputs; o++)
            {
                for (long i = 0; i < per; i++)
                {
                    w[o * per + i] *= scale[o];
                }
            }

            return HfTransforms.EncodeF32(w);
        });
        var bias = WeightInfo.Derived(name + ".bias", DataType.F32, new Shape(outputs), () =>
        {
            var scale = Scale();
            var b = HfTransforms.DecodeF32(beta);
            var m = HfTransforms.DecodeF32(mean);
            return HfTransforms.EncodeF32([.. b.Select((value, i) => value - m[i] * scale[i])]);
        });
        return (weight, bias);
    }
}
