using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Support check of the Mamba-2 scan: everything f32 (anything else is partitioned to the CPU provider).</summary>
    static OpSupport SupportsSelectiveScan(Node node)
    {
        foreach (var v in node.Inputs)
            if (v.Type != DataType.F32) return OpSupport.No($"SelectiveScan needs f32 inputs and state, got {v.Type} for {v.Name}");
        return OpSupport.Yes;
    }

    void LowerSelectiveScan(Node n)
    {
        var x = Ref(n.Inputs[0]); var dt = Ref(n.Inputs[1]); var dtb = Ref(n.Inputs[2]); var a = Ref(n.Inputs[3]);
        var b = Ref(n.Inputs[4]); var c = Ref(n.Inputs[5]); var d = Ref(n.Inputs[6]); var st = Ref(n.Inputs[7]);
        if (x.Strides[2] != 1) x = DenseTemp(x, "ssm-x");
        if (dt.Strides[1] != 1) dt = DenseTemp(dt, "ssm-dt");
        if (b.Strides[2] != 1) b = DenseTemp(b, "ssm-B");
        if (c.Strides[2] != 1) c = DenseTemp(c, "ssm-C");
        if (!dtb.IsDense) dtb = DenseTemp(dtb, "ssm-dtb");
        if (!a.IsDense) a = DenseTemp(a, "ssm-A");
        if (!d.IsDense) d = DenseTemp(d, "ssm-D");
        var y = Produce(n.Output);
        long H = x.Shape[1], P = x.Shape[2], G = b.Shape[1], N = b.Shape[2];
        // Rows of a head are independent: PB rows per workgroup, each owned by LPR lanes (lanes of a row stay inside one subgroup).
        int lpr = _opt.Has(DeviceFeatures.SubgroupOps) ? Math.Min(int.Parse(Environment.GetEnvironmentVariable("SHARPLIGHT_SSM_LPR") ?? "8"), _opt.SubgroupSize) : 1;
        while (N % lpr != 0 && lpr > 1) lpr /= 2;
        long pb = 16;
        while (pb > 1 && (P % pb != 0 || pb * lpr > 256)) pb /= 2;
        long tb = Math.Clamp(2048 / N, 1, 16);
        var k = Lib("ssm.slang", "ssm_scan", Macros(("H", H), ("P", P), ("G", G), ("N", N),
            ("XS0", x.Strides[0]), ("XS1", x.Strides[1]), ("DTS0", dt.Strides[0]), ("BS0", b.Strides[0]), ("BS1", b.Strides[1]), ("CS0", c.Strides[0]), ("CS1", c.Strides[1]),
            ("LPR", lpr), ("PB", pb), ("TB", tb), ("DT_MIN", n.Attributes.Get("dt_min", 0f))));
        Emit($"ssm_scan {n.Output.Name}", [n],
            [Always(k, _ => new Grid((uint)H, (uint)(P / pb)))],
            [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(dt.Buffer, dt.ByteOffset), new PointerArg(dtb.Buffer, dtb.ByteOffset), new PointerArg(a.Buffer, a.ByteOffset),
             new PointerArg(b.Buffer, b.ByteOffset), new PointerArg(c.Buffer, c.ByteOffset), new PointerArg(d.Buffer, d.ByteOffset), new PointerArg(st.Buffer, st.ByteOffset),
             new PointerArg(y.Buffer, y.ByteOffset), TokenArg(x)],
            [x, dt, dtb, a, b, c, d, st], [y, st]);
    }
}
