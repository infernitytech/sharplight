using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Import.HuggingFace;

/// <summary>Folds weight-normalized layers (<c>weight_g</c> and <c>weight_v</c>, as <c>torch.nn.utils.weight_norm</c> stores them) into plain weights.</summary>
internal static class HfWeightNormFolding
{
    /// <summary>
    /// The effective weight <c>g · v / ‖v‖</c> with the norm taken per row of the first dimension of <paramref name="direction"/>, computed on first read as F32.
    /// </summary>
    /// <param name="gain">The gain <c>g</c>, one value per row of <paramref name="direction"/>.</param>
    /// <param name="direction">The direction <c>v</c>.</param>
    /// <param name="name">The canonical name of the folded weight.</param>
    public static WeightInfo Fold(WeightInfo gain, WeightInfo direction, string name)
        => WeightInfo.Derived(name, DataType.F32, direction.Shape, () =>
        {
            var g = HfTransforms.DecodeF32(gain);
            var v = HfTransforms.DecodeF32(direction);
            long rows = direction.Shape[0].Fixed;
            long per = v.Length / rows;
            for (long r = 0; r < rows; r++)
            {
                double norm = 0;
                for (long i = 0; i < per; i++)
                {
                    norm += (double)v[(r * per) + i] * v[(r * per) + i];
                }

                float scale = (float)(g[r] / Math.Sqrt(norm));
                for (long i = 0; i < per; i++)
                {
                    v[(r * per) + i] *= scale;
                }
            }

            return HfTransforms.EncodeF32(v);
        });

    /// <summary>The element-wise sum of two same-shaped weights (the two bias vectors of an LSTM layer), computed on first read as F32.</summary>
    /// <param name="a">The first weight.</param>
    /// <param name="b">The second weight.</param>
    /// <param name="name">The canonical name of the sum.</param>
    public static WeightInfo Sum(WeightInfo a, WeightInfo b, string name)
        => WeightInfo.Derived(name, DataType.F32, a.Shape, () =>
        {
            var x = HfTransforms.DecodeF32(a);
            var y = HfTransforms.DecodeF32(b);
            for (int i = 0; i < x.Length; i++)
            {
                x[i] += y[i];
            }

            return HfTransforms.EncodeF32(x);
        });
}
