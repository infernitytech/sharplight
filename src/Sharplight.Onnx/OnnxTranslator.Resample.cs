using Sharplight.Graph;

namespace Sharplight.Onnx;

// Resize/Upsample. Plain 2-D nearest/bilinear resizes map onto the IR Resize op (a dedicated kernel on every provider). Everything else
// (cubic, antialias, exclude_outside, tf_crop_and_resize, half_pixel_symmetric, non-spatial axes, any rank, scales whose output size is
// rounded) is expressed as one constant-matrix multiplication per resized axis (ResizeAxis), which runs on the existing MatMul kernels.
internal sealed partial class OnnxTranslator
{
    private void Resize(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        int rank = x.Rank;
        bool upsample = n.OpType == "Upsample";
        bool legacyInputs = upsample || ins.Length == 2;   // opset < 11 Resize and Upsample carry scales as the second input
        var parameters = new ResizeParameters(
            Mode: ResizeMode(n),
            CoordinateMode: n.Str("coordinate_transformation_mode", legacyInputs ? "asymmetric" : "half_pixel"),
            NearestMode: n.Str("nearest_mode", upsample ? "floor" : "round_prefer_floor"),
            CubicCoeffA: n.Float("cubic_coeff_a", -0.75f),
            ExcludeOutside: n.Int("exclude_outside", 0) != 0,
            Antialias: n.Int("antialias", 0) != 0,
            ExtrapolationValue: n.Float("extrapolation_value", 0f));
        var (scales, sizes) = ResizeScalesAndSizes(n, ins, legacyInputs);
        int[] axes = (n.Ints("axes") ?? Enumerable.Range(0, rank).Select(a => (long)a).ToArray()).Select(a => NormalizeAxis(a, rank)).ToArray();
        var (outSize, scale) = ResizeTargets(x, axes, scales, sizes, n.Str("keep_aspect_ratio_policy", "stretch"));
        double[] roi = ResizeRoi(ins, legacyInputs, rank, axes, parameters.CoordinateMode);

        for (int a = 0; a < rank; a++)
        {
            bool changed = outSize[a] != (x.Shape[a].IsSymbolic ? -1 : x.Shape[a].Fixed) || Math.Abs(scale[a] - 1) > 1e-12 || roi[a] != 0 || roi[rank + a] != 1;
            if (changed && (x.Shape[a].IsSymbolic || !axes.Contains(a)))
            {
                throw new NotSupportedException(x.Shape[a].IsSymbolic ? "Resize of the symbolic batch axis" : "Resize changes an axis outside 'axes'");
            }
        }

        if (UsesResizeKernel(x, outSize, scale, parameters))
        {
            if (outSize[2] == x.Shape[2].Fixed && outSize[3] == x.Shape[3].Fixed && parameters.Mode == "nearest")
            {
                SetDyn(n, 0, x);
                return;
            }

            var attributes = Attributes.From(("size", new[] { outSize[2], outSize[3] }), ("mode", parameters.Mode), ("coord", parameters.CoordinateMode),
                ("nearest_mode", parameters.NearestMode));
            SetDyn(n, 0, Emit("Resize", [x], attributes, Nm(n)));
            return;
        }

        var y = x;
        foreach (int a in axes.Where(a => !x.Shape[a].IsSymbolic))
        {
            int inLen = (int)x.Shape[a].Fixed;
            if (outSize[a] == inLen && Math.Abs(scale[a] - 1) < 1e-12 && roi[a] == 0 && roi[rank + a] == 1)
            {
                continue;
            }

            var map = ResizeAxis.Build(inLen, (int)outSize[a], scale[a], parameters, roi[a], roi[rank + a]);
            y = ResampleAxis(y, a, inLen, (int)outSize[a], map, $"{Nm(n)}.axis{a}");
        }

        SetDyn(n, 0, y);
    }

    /// <summary>Whether the IR Resize kernel (2-D nearest/bilinear, scale = out / in) computes exactly this resize.</summary>
    private static bool UsesResizeKernel(Value x, long[] outSize, double[] scale, ResizeParameters p)
    {
        if (x.Rank != 4 || outSize[0] != (x.Shape[0].IsSymbolic ? -1 : x.Shape[0].Fixed) || outSize[1] != x.Shape[1].Fixed)
        {
            return false;
        }

        bool kernelMode = p.Mode is "nearest" or "linear" && p.CoordinateMode is "half_pixel" or "align_corners" or "asymmetric" or "pytorch_half_pixel"
            && !p.ExcludeOutside && p.NearestMode is "round_prefer_floor" or "round_prefer_ceil" or "floor" or "ceil";
        bool exactScales = Enumerable.Range(2, 2).All(a => Math.Abs(scale[a] - (double)outSize[a] / x.Shape[a].Fixed) < 1e-6 * scale[a]);
        bool noAntialias = !p.Antialias || p.Mode == "nearest" || (scale[2] >= 1 && scale[3] >= 1);
        return kernelMode && exactScales && noAntialias;
    }

    /// <summary>Applies one axis map: moves the axis last, multiplies by the constant [in, out] matrix (plus the extrapolation bias) and moves it back.</summary>
    private Value ResampleAxis(Value y, int axis, int inLen, int outLen, ResizeAxis map, string name)
    {
        int rank = y.Rank;
        var toLast = Enumerable.Range(0, rank).Where(a => a != axis).Append(axis).ToArray();
        var back = new int[rank];
        for (int k = 0; k < rank; k++)
        {
            back[toLast[k]] = k;
        }

        var transposed = new double[map.Weights.Length];
        for (int o = 0; o < outLen; o++)
        {
            for (int i = 0; i < inLen; i++)
            {
                transposed[(long)i * outLen + o] = map.Weights[(long)o * inLen + i];
            }
        }

        var moved = Permute(y, toLast);
        var flat = rank == 1 ? Rs(moved, 1, inLen) : moved;
        var r = MatMulWeightOrBatch(flat, OnnxConst.FromFloats(OnnxType.Float, [inLen, outLen], transposed), name);
        if (map.Bias is { } bias)
        {
            r = Emit("Add", [r, ConstValue(OnnxConst.FromFloats(OnnxType.Float, [outLen], bias), name + ".extrapolation")]);
        }

        return Permute(rank == 1 ? Rs(r, outLen) : r, back);
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
            sizes = OptConst(ins, 3, "Resize sizes", symbolic: true);
        }

        if (scales is { Count: 0 })
        {
            scales = null;
        }

        if (sizes is { Count: 0 })
        {
            sizes = null;
        }

        return (scales, sizes);
    }

    /// <summary>Per-axis output extent (-1 = the symbolic batch, unchanged) and scale factor, following the ONNX reference semantics.</summary>
    private static (long[] OutSize, double[] Scale) ResizeTargets(Value x, int[] axes, OnnxConst? scales, OnnxConst? sizes, string policy)
    {
        int rank = x.Rank;
        var inSize = x.Shape.Dims.Select(d => d.IsSymbolic ? -1L : d.Fixed).ToArray();
        var outSize = (long[])inSize.Clone();
        var scale = Enumerable.Repeat(1.0, rank).ToArray();
        if (sizes is not null)
        {
            var s = sizes.Ints();
            if (s.Length != axes.Length)
            {
                throw new InvalidDataException($"Resize sizes has {s.Length} entries for {axes.Length} axes");
            }

            for (int k = 0; k < axes.Length; k++)
            {
                int a = axes[k];
                if (SymbolicInts.IsSymbolic(s[k]))
                {
                    if (a != 0 || !x.Shape[0].IsSymbolic || s[k] != SymbolicInts.Encode(x.Shape[0]))
                    {
                        throw new NotSupportedException("Resize sizes depend on a symbolic size");
                    }

                    continue;
                }

                if (inSize[a] < 0)
                {
                    // A fixed size for the symbolic batch axis is the export-time batch; the batch is never resized.
                    continue;
                }

                outSize[a] = s[k];
                scale[a] = (double)s[k] / inSize[a];
            }

            if (policy != "stretch")
            {
                var chosen = axes.Where(a => inSize[a] >= 0).Select(a => scale[a]);
                double common = policy switch
                {
                    "not_larger" => chosen.Min(),
                    "not_smaller" => chosen.Max(),
                    _ => throw new NotSupportedException($"Resize keep_aspect_ratio_policy '{policy}'"),
                };
                foreach (int a in axes.Where(a => inSize[a] >= 0))
                {
                    scale[a] = common;
                    outSize[a] = (long)(common * inSize[a] + 0.5);
                }
            }

            return (outSize, scale);
        }

        if (scales is null)
        {
            throw new NotSupportedException("Resize needs constant scales or sizes");
        }

        var f = scales.Floats();
        if (f.Length != axes.Length)
        {
            throw new InvalidDataException($"Resize scales has {f.Length} entries for {axes.Length} axes");
        }

        for (int k = 0; k < axes.Length; k++)
        {
            int a = axes[k];
            double s = (float)f[k];
            if (inSize[a] < 0)
            {
                if (s != 1)
                {
                    throw new NotSupportedException("Resize of the symbolic batch axis");
                }

                continue;
            }

            scale[a] = s;
            outSize[a] = (long)MathF.Floor((float)s * inSize[a]);   // float32 product, as the ONNX reference and ORT compute it
        }

        return (outSize, scale);
    }

    /// <summary>The tf_crop_and_resize region as [starts..., ends...] over all axes (identity [0..., 1...] otherwise).</summary>
    private double[] ResizeRoi(TranslatedTensor?[] ins, bool legacyInputs, int rank, int[] axes, string coordinateMode)
    {
        var roi = Enumerable.Repeat(0.0, rank).Concat(Enumerable.Repeat(1.0, rank)).ToArray();
        if (legacyInputs || coordinateMode != "tf_crop_and_resize")
        {
            return roi;
        }

        var given = (OptConst(ins, 1, "Resize roi") ?? throw new NotSupportedException("tf_crop_and_resize Resize needs a roi")).Floats();
        if (given.Length != 2 * axes.Length)
        {
            throw new InvalidDataException($"Resize roi has {given.Length} entries for {axes.Length} axes");
        }

        for (int k = 0; k < axes.Length; k++)
        {
            roi[axes[k]] = given[k];
            roi[rank + axes[k]] = given[axes.Length + k];
        }

        return roi;
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
}
