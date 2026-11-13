using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

// Support checks for NCHW image ops, reductions and normalization.
internal static partial class CpuOpSupport
{
    private static OpSupport CheckSpatialAttributes(string op, Attributes a)
    {
        if (a.Has("strides") && (a.Get<long[]>("strides") is not { Length: 2 } strides || strides.Any(v => v < 1)))
        {
            return OpSupport.No($"{op}: strides must be 2 positive values");
        }

        if (a.Has("dilations") && (a.Get<long[]>("dilations") is not { Length: 2 } dilations || dilations.Any(v => v < 1)))
        {
            return OpSupport.No($"{op}: dilations must be 2 positive values");
        }

        if (a.Has("pads") && (a.Get<long[]>("pads") is not { Length: 4 } pads || pads.Any(v => v < 0)))
        {
            return OpSupport.No($"{op}: pads must be 4 non-negative values (top, left, bottom, right)");
        }

        return OpSupport.Yes;
    }

    private static OpSupport CheckConv(Node node)
    {
        var ins = node.Inputs;
        var input = RequireF32Input(node);
        if (!input.IsSupported)
        {
            return input;
        }

        var weight = ins[1];
        var weightSupport = CheckStaticWeight("Conv", weight);
        if (!weightSupport.IsSupported)
        {
            return weightSupport;
        }

        if (ins.Length > 2 && !F32Like(ins[2]))
        {
            return OpSupport.No($"Conv bias must be f32 (or a convertible weight), got {ins[2].Type}");
        }

        var a = node.Attributes;
        long groups = a.Get("groups", 1L);
        if (groups < 1 || ins[0].Shape.Dims[1].Fixed % groups != 0 || weight.Shape.Dims[0].Fixed % groups != 0)
        {
            return OpSupport.No($"Conv groups={groups} does not divide the channel counts");
        }

        if (ins[0].Shape.Dims[1].IsSymbolic || ins[0].Shape.Dims[2].IsSymbolic || ins[0].Shape.Dims[3].IsSymbolic)
        {
            return OpSupport.No("Conv with symbolic C/H/W");
        }

        return CheckSpatialAttributes("Conv", a);
    }

    private static OpSupport CheckConvTranspose(Node node)
    {
        var ins = node.Inputs;
        var input = RequireF32Input(node);
        if (!input.IsSupported)
        {
            return input;
        }

        var weightSupport = CheckStaticWeight("ConvTranspose", ins[1]);
        if (!weightSupport.IsSupported)
        {
            return weightSupport;
        }

        if (ins.Length > 2 && !F32Like(ins[2]))
        {
            return OpSupport.No($"ConvTranspose bias must be f32, got {ins[2].Type}");
        }

        long groups = node.Attributes.Get("groups", 1L);
        if (groups < 1 || ins[0].Shape.Dims[1].Fixed % groups != 0)
        {
            return OpSupport.No($"ConvTranspose groups={groups} does not divide the input channels");
        }

        return CheckSpatialAttributes("ConvTranspose", node.Attributes);
    }

    private static OpSupport CheckPool(Node node)
    {
        var input = RequireF32Input(node);
        if (!input.IsSupported)
        {
            return input;
        }

        var a = node.Attributes;
        string mode = a.Get("mode", "max");
        if (mode != "max" && mode != "avg")
        {
            return OpSupport.No($"Pool mode '{mode}' is not supported (max | avg)");
        }

        if (!a.Has("kernel") || a.Get<long[]>("kernel").Length != 2)
        {
            return OpSupport.No("Pool needs a 2-element kernel attribute");
        }

        if (node.Inputs[0].Shape.Dims[2].IsSymbolic || node.Inputs[0].Shape.Dims[3].IsSymbolic)
        {
            return OpSupport.No("Pool with symbolic H/W");
        }

        if (mode == "avg" && a.Get("dilations", new long[] { 1, 1 }).Any(d => d != 1))
        {
            return OpSupport.No("avg Pool with dilations != 1");
        }

        return CheckSpatialAttributes("Pool", a);
    }

    private static OpSupport CheckReduce(Node node)
    {
        var input = RequireF32Input(node);
        if (!input.IsSupported)
        {
            return input;
        }

        string mode = node.Attributes.Get("mode", "sum");
        if (mode is not ("sum" or "mean" or "max" or "min" or "prod" or "l1" or "l2" or "sumsquare" or "logsumexp"))
        {
            return OpSupport.No($"Reduce mode '{mode}' is not supported (sum | mean | max | min | prod | l1 | l2 | sumsquare | logsumexp)");
        }

        int rank = node.Inputs[0].Rank;
        foreach (var axis in node.Attributes.Get<long[]>("axes"))
        {
            if (axis < -rank || axis >= rank)
            {
                return OpSupport.No($"Reduce axis {axis} out of range for rank {rank}");
            }
        }

        return OpSupport.Yes;
    }

    private static OpSupport CheckPad(Node node)
    {
        var input = RequireF32Input(node);
        if (!input.IsSupported)
        {
            return input;
        }

        string mode = node.Attributes.Get("mode", "constant");
        return mode is "constant" or "reflect" or "edge"
            ? OpSupport.Yes
            : OpSupport.No($"Pad mode '{mode}' is not supported (constant | reflect | edge)");
    }

    private static OpSupport CheckResize(Node node)
    {
        var input = RequireF32Input(node);
        if (!input.IsSupported)
        {
            return input;
        }

        var a = node.Attributes;
        string mode = a.Get("mode", "nearest");
        string coordinateMode = a.Get("coord", "half_pixel");
        string nearestMode = a.Get("nearest_mode", "round_prefer_floor");
        if (mode is not ("nearest" or "linear"))
        {
            return OpSupport.No($"Resize mode '{mode}' is not supported (nearest | linear)");
        }

        if (coordinateMode is not ("half_pixel" or "align_corners" or "asymmetric" or "pytorch_half_pixel"))
        {
            return OpSupport.No($"Resize coord mode '{coordinateMode}' is not supported");
        }

        if (mode == "nearest" && nearestMode is not ("round_prefer_floor" or "round_prefer_ceil" or "floor" or "ceil"))
        {
            return OpSupport.No($"Resize nearest_mode '{nearestMode}' is not supported");
        }

        if (node.Inputs[0].Shape.Dims[2].IsSymbolic || node.Inputs[0].Shape.Dims[3].IsSymbolic)
        {
            return OpSupport.No("Resize with symbolic H/W");
        }

        return OpSupport.Yes;
    }

    private static OpSupport CheckGroupNorm(Node node)
    {
        var allF32 = AllF32(node);
        if (!allF32.IsSupported)
        {
            return allF32;
        }

        long groups = node.Attributes.Get<long>("groups");
        var channels = node.Inputs[0].Shape.Dims[1];
        return groups >= 1 && !channels.IsSymbolic && channels.Fixed % groups == 0
            ? OpSupport.Yes
            : OpSupport.No($"GroupNorm groups={groups} does not divide the channel count");
    }
}
