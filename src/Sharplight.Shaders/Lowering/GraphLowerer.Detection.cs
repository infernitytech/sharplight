using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Lowerings for detection ops: NonMaxSuppression, RoiAlign and max-pooling argmax indices (kernels in detection.slang), and NonZero
/// (nonzero.slang), whose data-dependent counts NonMaxSuppression usually consumes.
/// </summary>
public sealed partial class GraphLowerer
{
    private static string[] DetectionOpNames => ["NonMaxSuppression", "RoiAlign", "PoolArgMax", "NonZero"];
    private static readonly HashSet<string> s_detectionOps = [.. DetectionOpNames];

    /// <summary>Threads per workgroup of the NonMaxSuppression kernels (one bit-mask word per 32 candidates of a block).</summary>
    private const int NmsWorkgroup = 256;

    /// <summary>Shared memory of the per-class NonMaxSuppression kernel besides its sort slots: block boxes, ids, flags and masks.</summary>
    private const long NmsFixedSharedBytes = NmsWorkgroup * (16 + 4 + 4) + NmsWorkgroup * (NmsWorkgroup / 32) * 4 + 16;

    private static OpSupport SupportsDetection(Node node)
    {
        switch (node.Op.Name)
        {
            case "NonMaxSuppression":
                if (node.Inputs[0].Type != DataType.F32 || node.Inputs[1].Type != DataType.F32)
                {
                    return OpSupport.No("NonMaxSuppression expects f32 boxes and scores");
                }

                return node.Inputs[1].Rank is 1 or 3 ? OpSupport.Yes : OpSupport.No("NonMaxSuppression scores must be [S] or [B, C, S]");
            case "RoiAlign":
                if (node.Inputs[0].Type != DataType.F32 || node.Inputs[1].Type != DataType.F32 || node.Inputs[2].Type != DataType.I32)
                {
                    return OpSupport.No("RoiAlign expects f32 x and rois and i32 batch indices");
                }

                return node.Inputs[0].Rank == 4 ? OpSupport.Yes : OpSupport.No("RoiAlign expects a 4-D feature map");
            case "NonZero":
                return node.Inputs[0].Type is { IsQuantized: false, Kind: not DataTypeKind.String } x && x.ElementSize is 1 or 2 or 4 or 8 && node.Inputs[0].Rank >= 1
                    ? OpSupport.Yes
                    : OpSupport.No($"NonZero of {node.Inputs[0].Type}");
            default:
                return node.Inputs[0].Type is { IsQuantized: false } t && (t == DataType.F32 || t == DataType.F16 || t == DataType.BF16) && node.Inputs[0].Rank == 4
                    ? OpSupport.Yes
                    : OpSupport.No("PoolArgMax expects a 4-D float input");
        }
    }

    private bool LowerDetection(Node n)
    {
        switch (n.Op.Name)
        {
            case "NonMaxSuppression":
                LowerNonMaxSuppression(n);
                return true;
            case "RoiAlign":
                LowerRoiAlign(n);
                return true;
            case "PoolArgMax":
                LowerPoolArgMax(n);
                return true;
            case "NonZero":
                LowerNonZero(n);
                return true;
        }

        return false;
    }

    /// <summary>
    /// NonMaxSuppression in two launches: one workgroup per (batch, class) sorts its candidates and suppresses greedily, then one
    /// workgroup concatenates the kept boxes into the output rows and writes their number, the op's data-dependent count.
    /// </summary>
    private void LowerNonMaxSuppression(Node n)
    {
        var a = n.Attributes;
        var boxes = AsDenseF32(Ref(n.Inputs[0]), "nms-boxes");
        var scores = AsDenseF32(Ref(n.Inputs[1]), "nms-scores");
        var y = Produce(n.Output);
        var count = ProduceSymbol(a.Get<string>("count_symbol"));
        bool single = scores.Rank == 1;   // boxes [S, 4], scores [S]: one batch and class
        long classes = single ? 1 : scores.Shape[1];
        long boxesBound = single ? scores.Shape[0] : scores.Shape[2];
        long groupsBound = single ? 1 : scores.Shape[0] * classes;
        long maxKept = Math.Max(0, Math.Min(a.Get<long>("max_per_class"), boxesBound));
        long keptSlots = Math.Max(1, maxKept);
        long sortSlots = 2;
        while (sortSlots < boxesBound)
        {
            sortSlots <<= 1;
        }

        // Shared memory: the kept boxes when they fit next to a 1024-slot sort, then as many sort slots as fit.
        bool keptLocal = NmsFixedSharedBytes + 16 * keptSlots + 8 * Math.Min(sortSlots, 1024) <= _opt.MaxSharedMemory;
        long sharedBytes = NmsFixedSharedBytes + (keptLocal ? 16 * keptSlots : 0);
        long localSlots = 1;
        while (sharedBytes + 8 * localSlots * 2 <= _opt.MaxSharedMemory && localSlots * 2 <= sortSlots)
        {
            localSlots <<= 1;
        }

        var keys = Temp(DataType.F32, [groupsBound * sortSlots], null);
        var ids = Temp(DataType.I32, [groupsBound * sortSlots], null);
        var keptIds = Temp(DataType.I32, [groupsBound * keptSlots], null);
        var keptBoxes = Temp(DataType.F32, [groupsBound * keptSlots * 4], null);
        var keptCount = Temp(DataType.I32, [groupsBound], null);
        var offsets = Temp(DataType.I32, [groupsBound], null);

        var classKernel = Lib("detection.slang", "nms_class", Macros(("WG", NmsWorkgroup), ("C", classes), ("MAXK", maxKept), ("KCAP", keptSlots),
            ("CAP", sortSlots), ("LOCAL", localSlots), ("KEPT_LOCAL", keptLocal), ("IOU", a.Get("iou_threshold", 0f)), ("SCORE", a.Get("score_threshold", float.NegativeInfinity)),
            ("CENTER", a.Get("center_point_box", false))));
        KernelArg groups = single || scores.Symbol is null ? new UIntArg((uint)groupsBound) : new SymbolArg(scores.Symbol, classes);
        KernelArg boxCount = single && scores.Symbol is not null ? new SymbolArg(scores.Symbol, 1) : new UIntArg((uint)boxesBound);
        var scoreRef = scores;
        Emit($"nms {n.Output.Name}", [n], [Always(classKernel, b => TopKGrid(single ? 1 : Rows(b, scoreRef) * classes))],
            [Pointer(boxes), Pointer(scores), Pointer(keys), Pointer(ids), Pointer(keptIds), Pointer(keptBoxes), Pointer(keptCount), groups, boxCount],
            [boxes, scores], [keys, ids, keptIds, keptBoxes, keptCount]);

        var gatherKernel = Lib("detection.slang", "nms_gather", Macros(("WG", NmsWorkgroup), ("C", classes), ("KCAP", keptSlots)));
        Emit($"nms_gather {n.Output.Name}", [n], [Always(gatherKernel, _ => new Grid(1))],
            [Pointer(keptIds), Pointer(keptCount), Pointer(offsets), Pointer(y), Pointer(count), groups],
            [keptIds, keptCount], [offsets, y, count]);
    }

    /// <summary>NonZero as one workgroup (an ordered compaction) that also writes the number of rows, the op's data-dependent count.</summary>
    private void LowerNonZero(Node n)
    {
        var x = EnsureDense(Ref(n.Inputs[0]), "nonzero");
        var y = Produce(n.Output);
        var count = ProduceSymbol(n.Attributes.Get<string>("count_symbol"));
        bool isFloat = x.Type == DataType.F32 || x.Type == DataType.F16 || x.Type == DataType.BF16;
        string mask = x.Type.ElementSize switch
        {
            1 => "0xFFu",
            2 => isFloat ? "0x7FFFu" : "0xFFFFu",
            4 => isFloat ? "0x7FFFFFFFu" : "0xFFFFFFFFu",
            _ => "0xFFFFFFFFFFFFFFFFull",
        };
        string inner = x.Rank == 1 ? "1" : string.Join(",", x.Shape.Skip(1));
        var kernel = Lib("nonzero.slang", "nonzero", Macros(("RAW", RawType(x.Type)), ("MASK", mask), ("RANK", x.Rank), ("DIMS", inner), ("WG", 256)));
        Emit($"nonzero {n.Output.Name}", [n], [Always(kernel, _ => new Grid(1))], [Pointer(x), Pointer(y), Pointer(count), CountArg(x, x.MaxElements / x.Shape[0])],
            [x], [y, count]);
    }

    private void LowerRoiAlign(Node n)
    {
        var a = n.Attributes;
        var x = AsF32(Ref(n.Inputs[0]), "roialign");
        var rois = AsDenseF32(Ref(n.Inputs[1]), "roialign-rois");
        var batch = EnsureDense(Ref(n.Inputs[2]), "roialign-batch");
        var y = Produce(n.Output);
        long channels = x.Shape[1];
        long outH = y.Shape[2];
        long outW = y.Shape[3];
        var kernel = Lib("detection.slang", "roi_align", Macros(("C", channels), ("H", x.Shape[2]), ("W", x.Shape[3]), ("OH", outH), ("OW", outW),
            ("XS_N", x.Strides[0]), ("XS_C", x.Strides[1]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]),
            ("RATIO", a.Get("sampling_ratio", 0L)), ("SCALE", a.Get("spatial_scale", 1f)), ("MODE_MAX", a.Get("mode", "avg") == "max"),
            ("HALF_PIXEL", a.Get("half_pixel", true))));
        long perRoi = channels * outH * outW;
        var roiRef = rois;
        Emit($"roi_align {n.Output.Name}", [n], [Always(kernel, b => new Grid(Grid.Div(Rows(b, roiRef) * perRoi, 256)))],
            [Pointer(x), Pointer(rois), Pointer(batch), Pointer(y), CountArg(rois, perRoi)], [x, rois, batch], [y]);
    }

    private void LowerPoolArgMax(Node n)
    {
        var a = n.Attributes;
        var x = AsF32(Ref(n.Inputs[0]), "pool-argmax");
        var y = Produce(n.Output);
        var kernel = a.Get<long[]>("kernel");
        var strides = a.Get("strides", new long[] { 1, 1 });
        var pads = a.Get("pads", new long[] { 0, 0, 0, 0 });
        var dilations = a.Get("dilations", new long[] { 1, 1 });
        long channels = x.Shape[1];
        long outH = y.Shape[2];
        long outW = y.Shape[3];
        var k = Lib("detection.slang", "pool_argmax", Macros(("C", channels), ("H", x.Shape[2]), ("W", x.Shape[3]), ("HO", outH), ("WO", outW),
            ("KH", kernel[0]), ("KW", kernel[1]), ("SH", strides[0]), ("SW", strides[1]), ("PH", pads[0]), ("PW", pads[1]), ("DH", dilations[0]), ("DW", dilations[1]),
            ("XS_N", x.Strides[0]), ("XS_C", x.Strides[1]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]), ("COLUMN_MAJOR", a.Get("storage_order", 0L) == 1)));
        var xRef = x;
        Emit($"pool_argmax {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, xRef) * channels * outH * outW, 256)))],
            [Pointer(x), Pointer(y), CountArg(x, channels * outH * outW)], [x], [y]);
    }
}
