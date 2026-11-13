using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Separable resampling expressed with existing ops: Resize modes without a dedicated kernel (cubic) become two constant-matrix
// multiplications (rows = output positions, columns = input positions), one per spatial axis.
internal sealed partial class OnnxTranslator
{
    /// <summary>Interpolation matrix [outLen, inLen] of ONNX Resize mode "cubic" (border indices are clamped, optionally excluded).</summary>
    static double[] CubicMatrix(int inLen, int outLen, double scale, string coord, double a, bool excludeOutside)
    {
        var m = new double[(long)outLen * inLen];
        for (int o = 0; o < outLen; o++)
        {
            double x = coord switch
            {
                "half_pixel" => (o + 0.5) / scale - 0.5,
                "pytorch_half_pixel" => outLen > 1 ? (o + 0.5) / scale - 0.5 : 0,
                "align_corners" => outLen == 1 ? 0 : o * (double)(inLen - 1) / (outLen - 1),
                "asymmetric" => o / scale,
                _ => throw new NotSupportedException($"Resize coordinate_transformation_mode '{coord}' for cubic"),
            };
            double fl = Math.Floor(x), s = x - fl;
            double[] w =
            [
                ((a * (s + 1) - 5 * a) * (s + 1) + 8 * a) * (s + 1) - 4 * a,
                ((a + 2) * s - (a + 3)) * s * s + 1,
                ((a + 2) * (1 - s) - (a + 3)) * (1 - s) * (1 - s) + 1,
                ((a * (2 - s) - 5 * a) * (2 - s) + 8 * a) * (2 - s) - 4 * a,
            ];
            if (excludeOutside)
            {
                double sum = 0;
                for (int k = 0; k < 4; k++) { long i = (long)fl - 1 + k; if (i < 0 || i >= inLen) w[k] = 0; sum += w[k]; }
                if (sum != 0) for (int k = 0; k < 4; k++) w[k] /= sum;
            }
            for (int k = 0; k < 4; k++)
            {
                long i = Math.Clamp((long)fl - 1 + k, 0, inLen - 1);
                m[(long)o * inLen + i] += w[k];
            }
        }
        return m;
    }

    Value ResizeCubic(OnnxNode n, Value x, long[] size, double[]? scaleHW, string coord)
    {
        int h = (int)x.Shape[2].Fixed, w = (int)x.Shape[3].Fixed;
        double a = n.Float("cubic_coeff_a", -0.75f);
        bool excl = n.Int("exclude_outside", 0) != 0;
        string name = Nm(n);
        double sh = scaleHW?[0] ?? (double)size[0] / h, sw = scaleHW?[1] ?? (double)size[1] / w;
        // out[..., o] = sum_i x[..., i] * M[o, i]  ==  x @ M^T ; the weight cache wants [K, N] = M^T.
        OnnxConst Mt(double[] m, int outLen, int inLen)
        {
            var t = new double[m.Length];
            for (int o = 0; o < outLen; o++) for (int i = 0; i < inLen; i++) t[(long)i * outLen + o] = m[(long)o * inLen + i];
            return OnnxConst.FromFloats(OnnxType.Float, [inLen, outLen], t);
        }
        var y = x;
        if (size[1] != w || sw != 1)
            y = MatMulWeightOrBatch(y, Mt(CubicMatrix(w, (int)size[1], sw, coord, a, excl), (int)size[1], w), name + ".w");
        if (size[0] != h || sh != 1)
        {
            var p = Permute(y, [0, 1, 3, 2]);
            p = MatMulWeightOrBatch(p, Mt(CubicMatrix(h, (int)size[0], sh, coord, a, excl), (int)size[0], h), name + ".h");
            y = Permute(p, [0, 1, 3, 2]);
        }
        return y;
    }
}
