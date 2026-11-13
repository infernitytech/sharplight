using Sharplight.Graph;

namespace Sharplight.Onnx;

// Resize/Upsample. Modes without a dedicated kernel (cubic) are expressed with existing ops as two constant-matrix multiplications
// (rows = output positions, columns = input positions), one per spatial axis.
internal sealed partial class OnnxTranslator
{
    private void Resize(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        if (x.Rank != 4)
        {
            throw new NotSupportedException("only 4-D (NCHW) Resize is supported");
        }

        bool upsample = n.OpType == "Upsample";
        bool legacyInputs = upsample || ins.Length == 2;   // opset < 11 Resize and Upsample carry scales as the second input
        var (scales, sizes) = ResizeScalesAndSizes(n, ins, legacyInputs);
        long h = x.Shape[2].Fixed;
        long w = x.Shape[3].Fixed;
        long[] size = ResizeOutputSize(x, scales, sizes);

        string mode = ResizeMode(n);
        string coord = n.Str("coordinate_transformation_mode", legacyInputs ? "asymmetric" : "half_pixel");
        if (coord is not ("half_pixel" or "align_corners" or "asymmetric" or "pytorch_half_pixel"))
        {
            throw new NotSupportedException($"Resize coordinate_transformation_mode '{coord}'");
        }

        if (n.Int("antialias", 0) != 0)
        {
            throw new NotSupportedException("antialiased Resize");
        }

        if (n.Int("exclude_outside", 0) != 0)
        {
            throw new NotSupportedException("Resize exclude_outside");
        }

        string nearest = n.Str("nearest_mode", upsample ? "floor" : "round_prefer_floor");
        if (mode == "cubic")
        {
            double[]? scaleHW = scales is not null && sizes is null ? [scales.Floats()[2], scales.Floats()[3]] : null;
            SetDyn(n, 0, ResizeCubic(n, x, size, scaleHW, coord));
            return;
        }

        if (size[0] == h && size[1] == w && mode == "nearest")
        {
            SetDyn(n, 0, x);
            return;
        }

        var attributes = Attributes.From(("size", size), ("mode", mode), ("coord", coord), ("nearest_mode", nearest));
        SetDyn(n, 0, Emit("Resize", [x], attributes, Nm(n)));
    }

    private (OnnxConst? Scales, OnnxConst? Sizes) ResizeScalesAndSizes(OnnxNode n, TranslatedTensor?[] ins, bool legacyInputs)
    {
        OnnxConst? scales;
        OnnxConst? sizes = null;
        if (legacyInputs)
        {
            var scaleAttribute = n.Floats("scales");
            scales = n.Has("scales")
                ? OnnxConst.FromFloats(OnnxType.Float, [scaleAttribute!.Length], scaleAttribute.Select(v => (double)v).ToArray())
                : OptConst(ins, 1, "Resize scales");
        }
        else
        {
            scales = OptConst(ins, 2, "Resize scales");
            sizes = OptConst(ins, 3, "Resize sizes");
        }

        if (scales is { Count: 0 })
        {
            scales = null;
        }

        return (scales, sizes);
    }

    /// <summary>Output [height, width]; only the spatial dimensions may change.</summary>
    private static long[] ResizeOutputSize(Value x, OnnxConst? scales, OnnxConst? sizes)
    {
        if (sizes is not null)
        {
            var s = sizes.Ints();
            if (s.Length != 4 || s[1] != x.Shape[1].Fixed || (!Sym(x) && s[0] != x.Shape[0].Fixed))
            {
                throw new NotSupportedException("Resize may only change the spatial dimensions");
            }

            return [s[2], s[3]];
        }

        if (scales is not null)
        {
            var s = scales.Floats();
            if (s.Length != 4 || s[0] != 1 || s[1] != 1)
            {
                throw new NotSupportedException("Resize may only scale the spatial dimensions");
            }

            float h = x.Shape[2].Fixed;
            float w = x.Shape[3].Fixed;
            return [(long)MathF.Floor(h * (float)s[2]), (long)MathF.Floor(w * (float)s[3])];
        }

        throw new NotSupportedException("Resize needs constant scales or sizes");
    }

    private static string ResizeMode(OnnxNode n)
    {
        string mode = n.Str("mode", "nearest");
        mode = mode switch
        {
            "bilinear" => "linear",
            "bicubic" => "cubic",
            _ => mode,
        };
        if (mode is not ("nearest" or "linear" or "cubic"))
        {
            throw new NotSupportedException($"Resize mode '{mode}'");
        }

        return mode;
    }

    /// <summary>Interpolation matrix [outLen, inLen] of ONNX Resize mode "cubic" (border indices are clamped, optionally excluded).</summary>
    private static double[] CubicMatrix(int inLen, int outLen, double scale, string coord, double a, bool excludeOutside)
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
            double fl = Math.Floor(x);
            double s = x - fl;
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
                for (int k = 0; k < 4; k++)
                {
                    long i = (long)fl - 1 + k;
                    if (i < 0 || i >= inLen)
                    {
                        w[k] = 0;
                    }
                    sum += w[k];
                }
                if (sum != 0)
                {
                    for (int k = 0; k < 4; k++)
                    {
                        w[k] /= sum;
                    }
                }
            }
            for (int k = 0; k < 4; k++)
            {
                long i = Math.Clamp((long)fl - 1 + k, 0, inLen - 1);
                m[(long)o * inLen + i] += w[k];
            }
        }
        return m;
    }

    private Value ResizeCubic(OnnxNode n, Value x, long[] size, double[]? scaleHW, string coord)
    {
        int h = (int)x.Shape[2].Fixed;
        int w = (int)x.Shape[3].Fixed;
        double a = n.Float("cubic_coeff_a", -0.75f);
        bool excl = n.Int("exclude_outside", 0) != 0;
        string name = Nm(n);
        double sh = scaleHW?[0] ?? (double)size[0] / h;
        double sw = scaleHW?[1] ?? (double)size[1] / w;
        // out[..., o] = sum_i x[..., i] * M[o, i]  ==  x @ M^T ; the weight cache wants [K, N] = M^T.
        OnnxConst Mt(double[] m, int outLen, int inLen)
        {
            var t = new double[m.Length];
            for (int o = 0; o < outLen; o++)
            {
                for (int i = 0; i < inLen; i++)
                {
                    t[(long)i * outLen + o] = m[(long)o * inLen + i];
                }
            }

            return OnnxConst.FromFloats(OnnxType.Float, [inLen, outLen], t);
        }
        var y = x;
        if (size[1] != w || sw != 1)
        {
            y = MatMulWeightOrBatch(y, Mt(CubicMatrix(w, (int)size[1], sw, coord, a, excl), (int)size[1], w), name + ".w");
        }

        if (size[0] != h || sh != 1)
        {
            var p = Permute(y, [0, 1, 3, 2]);
            p = MatMulWeightOrBatch(p, Mt(CubicMatrix(h, (int)size[0], sh, coord, a, excl), (int)size[0], h), name + ".h");
            y = Permute(p, [0, 1, 3, 2]);
        }
        return y;
    }
}
