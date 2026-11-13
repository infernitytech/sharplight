using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Support check of the Mamba-2 scan: everything f32 (anything else is partitioned to the CPU provider).</summary>
    private static OpSupport SupportsSelectiveScan(Node node)
    {
        foreach (var v in node.Inputs.Take(8))
        {
            if (v.Type != DataType.F32)
            {
                return OpSupport.No($"SelectiveScan needs f32 inputs and state, got {v.Type} for {v.Name}");
            }
        }

        return OpSupport.Yes;
    }

    /// <summary>Mamba-2 selective scan: one workgroup per (head, block of state rows); the state is updated in place.</summary>
    private void LowerSelectiveScan(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var dt = Ref(n.Inputs[1]);
        var dtb = Ref(n.Inputs[2]);
        var a = Ref(n.Inputs[3]);
        var b = Ref(n.Inputs[4]);
        var c = Ref(n.Inputs[5]);
        var d = Ref(n.Inputs[6]);
        var st = Ref(n.Inputs[7]);
        var runs = n.Inputs.Length > 8 ? DenseInput(n.Inputs[8], "seq-runs") : null;
        x = EnsureUnitInnerStride(x, 2, "ssm-x");
        dt = EnsureUnitInnerStride(dt, 1, "ssm-dt");
        b = EnsureUnitInnerStride(b, 2, "ssm-B");
        c = EnsureUnitInnerStride(c, 2, "ssm-C");
        dtb = EnsureDense(dtb, "ssm-dtb");
        a = EnsureDense(a, "ssm-A");
        d = EnsureDense(d, "ssm-D");

        var y = Produce(n.Output);
        long H = x.Shape[1];
        long P = x.Shape[2];
        long G = b.Shape[1];
        long N = b.Shape[2];
        // Rows of a head are independent: PB rows per workgroup, each owned by LPR lanes (lanes of a row stay inside one subgroup).
        int lanesPerRow = _opt.Has(DeviceFeatures.SubgroupOps) ? Math.Min(DiagnosticSwitches.SsmLanesPerRow, _opt.SubgroupSize) : 1;
        while (N % lanesPerRow != 0 && lanesPerRow > 1)
        {
            lanesPerRow /= 2;
        }

        long rowsPerBlock = 16;
        while (rowsPerBlock > 1 && (P % rowsPerBlock != 0 || rowsPerBlock * lanesPerRow > 256))
        {
            rowsPerBlock /= 2;
        }

        long timeBlock = Math.Clamp(2048 / N, 1, 16);
        var k = Lib("ssm.slang", "ssm_scan", Macros(("H", H), ("P", P), ("G", G), ("N", N),
            ("XS0", x.Strides[0]), ("XS1", x.Strides[1]), ("DTS0", dt.Strides[0]), ("BS0", b.Strides[0]), ("BS1", b.Strides[1]), ("CS0", c.Strides[0]), ("CS1", c.Strides[1]),
            ("LPR", lanesPerRow), ("PB", rowsPerBlock), ("TB", timeBlock), ("DT_MIN", n.Attributes.Get("dt_min", 0f)), ("MULTI_SEQ", runs is not null)));
        Emit($"ssm_scan {n.Output.Name}", [n],
            [Always(k, b => new Grid((uint)H, (uint)(P / rowsPerBlock), runs is null ? 1u : (uint)Rows(b, runs)))],
            [Pointer(x), Pointer(dt), Pointer(dtb), Pointer(a),
             Pointer(b), Pointer(c), Pointer(d), Pointer(st),
             Pointer(y), .. SeqArgs(runs), TokenArg(x)],
            [x, dt, dtb, a, b, c, d, st, .. runs is null ? [] : (TensorRef[])[runs]], [y, st]);
    }
}
