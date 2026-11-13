using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    // ---- recurrent ---------------------------------------------------------------------------
    private void LowerConv(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var w = Ref(n.Inputs[1]);
        var st = Ref(n.Inputs[2]);
        if (x.Rank != 2 || x.Strides[1] != 1)
        {
            x = DenseTemp(x, "conv");
        }

        if (w.Type != DataType.F32)
        {
            throw new NotSupportedException("conv weight must be f32");
        }

        var y = Produce(n.Output);
        long C = x.Shape[1];
        long KS = w.Shape[1];
        // Optional bias (attribute "bias", or a non-i32 input 3) and, in the multi-sequence form, the trailing i32 sequence-runs table.
        bool hasBias = n.Attributes.Get("bias", false) || (n.Inputs.Length > 3 && n.Inputs[3].Type != DataType.I32);
        TensorRef? cb = hasBias ? Ref(n.Inputs[3]) : null;
        var runs = n.Inputs.Length > (hasBias ? 4 : 3) ? DenseInput(n.Inputs[^1], "seq-runs") : null;
        var k = Lib("recurrent.slang", "conv1d", Macros(("C", C), ("KS", KS), ("XS", x.Strides[0]), ("ACT_SILU", n.Attributes.Get("activation", "none") == "silu"),
            ("HAS_BIAS", cb is not null), ("MULTI_SEQ", runs is not null)));
        Emit($"conv1d {n.Output.Name}", [n],
            [Always(k, b => new Grid(Grid.Div(C, 128), 1, runs is null ? 1u : (uint)Rows(b, runs)))],
            [Pointer(x), Pointer(w), Pointer(st), Pointer(y),
             Pointer(cb ?? y), .. SeqArgs(runs), TokenArg(x)],
            [x, w, st, .. cb is null ? [] : (TensorRef[])[cb], .. runs is null ? [] : (TensorRef[])[runs]], [y, st]);
    }

    private void LowerDeltaRule(Node n)
    {
        // q/k may come straight from an absorbed L2Norm: read its input and normalize in the kernel.
        Node? nq = _absorbedNorms.GetValueOrDefault(n.Inputs[0]), nk = _absorbedNorms.GetValueOrDefault(n.Inputs[1]);
        var q = Ref(nq?.Inputs[0] ?? n.Inputs[0]);
        var k = Ref(nk?.Inputs[0] ?? n.Inputs[1]);
        var v = Ref(n.Inputs[2]);
        var g = Ref(n.Inputs[3]);
        var beta = Ref(n.Inputs[4]);
        var st = Ref(n.Inputs[5]);
        var runs = n.Inputs.Length > 6 ? DenseInput(n.Inputs[6], "seq-runs") : null;
        if (q.Strides[2] != 1)
        {
            q = DenseTemp(q, "delta-q");
        }

        if (k.Strides[2] != 1)
        {
            k = DenseTemp(k, "delta-k");
        }

        if (v.Strides[2] != 1)
        {
            v = DenseTemp(v, "delta-v");
        }

        if (!g.IsDense)
        {
            g = DenseTemp(g, "delta-g");
        }

        if (!beta.IsDense)
        {
            beta = DenseTemp(beta, "delta-beta");
        }

        var o = Produce(n.Output);
        long HK = q.Shape[1];
        long DK = q.Shape[2];
        long HV = v.Shape[1];
        long DV = v.Shape[2];
        int lpr = DiagnosticSwitches.DeltaLanesPerRow;
        lpr = Math.Min(lpr, _opt.SubgroupSize);
        while ((DV * lpr > 1024 || DK % lpr != 0) && lpr > 1)
        {
            lpr /= 2;
        }
        // Split each head's DV state rows over several workgroups (rows are independent) for occupancy.
        long dvbWant = DiagnosticSwitches.DeltaRowsPerBlock;
        long dvb = DV % dvbWant == 0 ? dvbWant : DV;
        var kern = Lib("recurrent.slang", "delta_rule", Macros(("HK", HK), ("HV", HV), ("DK", DK), ("DV", DV),
            ("QS0", q.Strides[0]), ("QS1", q.Strides[1]), ("KS0", k.Strides[0]), ("KS1", k.Strides[1]), ("VS0", v.Strides[0]), ("VS1", v.Strides[1]),
            ("SCALE", n.Attributes.Get<float>("scale")), ("LPR", lpr), ("DVB", dvb), ("TB", Math.Clamp(2048 / DK, 1, 16)),
            ("NORM_Q", nq is not null), ("EPS_Q", nq?.Attributes.Get<float>("eps") ?? 0f), ("NORM_K", nk is not null), ("EPS_K", nk?.Attributes.Get<float>("eps") ?? 0f),
            ("MULTI_SEQ", runs is not null)));
        Emit($"delta_rule {n.Output.Name}", [.. new[] { nq, nk }.OfType<Node>(), n],
            [Always(kern, b => new Grid((uint)HV, (uint)(DV / dvb), runs is null ? 1u : (uint)Rows(b, runs)))],
            [Pointer(q), Pointer(k), Pointer(v),
             Pointer(g), Pointer(beta), Pointer(st),
             Pointer(o), .. SeqArgs(runs), TokenArg(q)],
            [q, k, v, g, beta, st, .. runs is null ? [] : (TensorRef[])[runs]], [o, st]);
    }
}
