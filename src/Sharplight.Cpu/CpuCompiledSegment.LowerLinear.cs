using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

// Matrix products and embedding lookups; weights stay in their (possibly quantized) storage format.
internal sealed unsafe partial class CpuCompiledSegment
{
    /// <summary>The value as a weight operand: weights/constants are used in place, activations must be dense.</summary>
    private TRef WeightOperand(Value v)
    {
        var r = Ref(v);
        return IsStatic(v) ? r : Dense(r);
    }

    private void LowerGather(Node n)
    {
        var ins = n.Inputs;
        var table = Ref(ins[0]);
        var indices = Ref(ins[1]);
        var y = NewDense(n.Output);
        var format = Quant.FormatOf(table.Type)!.Value;
        if (!IsStatic(ins[0]))
        {
            table = Dense(table);
        }

        AddStep(new GatherOp(table, indices, y, format), table, indices, y);
        _refs[n.Output] = y;
    }

    private void LowerMatMul(Node n)
    {
        var ins = n.Inputs;
        var x = RowContig(RefF32(ins[0]));
        var w = WeightOperand(ins[1]);
        var format = Quant.FormatOf(w.Type)!.Value;
        var y = NewDense(n.Output);
        bool intDot = _features.Contains(DeviceFeatures.IntDot) && (_options.AllowReducedPrecision || !n.Attributes.Get(StandardOps.F32ActivationsAttribute, false));
        var epilogue = AbsorbEpilogue(n.Output, y, columnBias: true);
        AddStep(new MatMulOp(x, w, y, format, intDot, IsStatic(ins[1]), epilogue), Touched(x, w, y, epilogue?.ColumnBias, epilogue?.Residual));
        _refs[n.Output] = y;
    }

    /// <summary>Mixture-of-experts matmul: the weight is [experts, N, K] and a per-row index selects the expert.</summary>
    private void LowerMatMulId(Node n)
    {
        var ins = n.Inputs;
        var x = RowContig(RefF32(ins[0]));
        var w = Ref(ins[1]);
        var expertIndex = Dense(Ref(ins[2]));
        if (!IsStatic(ins[1]))
        {
            w = Dense(w);
        }

        var format = Quant.FormatOf(w.Type)!.Value;
        var y = NewDense(n.Output);
        bool intDot = _features.Contains(DeviceFeatures.IntDot) && (_options.AllowReducedPrecision || !n.Attributes.Get(StandardOps.F32ActivationsAttribute, false));
        if (IsStatic(ins[1]) && PairedMatMulId(n) is { } pair)
        {
            // Gate and up projections of the experts: one pass over both, with the gated activation SiLU(gate) * up when it follows.
            var w2 = Ref(pair.Inputs[1]);
            var y2 = NewDense(pair.Output);
            if (GatedActivation(n, pair) is { } gated)
            {
                bool gateFirst = gated.Gate == n;
                var h = NewDense(gated.Mul.Output);
                var (gateWeight, gateOut, upWeight, upOut) = gateFirst ? (w, y, w2, y2) : (w2, y2, w, y);
                AddStep(new MatMulIdOp(x, gateWeight, expertIndex, gateOut, format, intDot, upWeight, upOut, h), x, w, expertIndex, y, w2, y2, h);
                _refs[gated.Silu.Output] = gateOut;
                _refs[gated.Mul.Output] = h;
                _absorbed.Add(gated.Silu);
                _absorbed.Add(gated.Mul);
            }
            else
            {
                AddStep(new MatMulIdOp(x, w, expertIndex, y, format, intDot, w2, y2), x, w, expertIndex, y, w2, y2);
            }

            _refs[n.Output] = y;
            _refs[pair.Output] = y2;
            _absorbed.Add(pair);
            return;
        }

        AddStep(new MatMulIdOp(x, w, expertIndex, y, format, intDot), x, w, expertIndex, y);
        _refs[n.Output] = y;
    }

    /// <summary>
    /// The MLP gating of an expert pair <paramref name="a"/> / <paramref name="b"/>: <c>Mul(Silu(gate), up)</c> (either operand order), each
    /// product read only by it and the SiLU only by the Mul, with the products' shape; null when the pair is not followed by it.
    /// </summary>
    private (Node Gate, Node Silu, Node Mul)? GatedActivation(Node a, Node b)
    {
        foreach (var (gate, up) in new[] { (a, b), (b, a) })
        {
            if (SoleConsumer(gate.Output) is { Op.Name: "Silu" } silu && SoleConsumer(silu.Output) is { Op.Name: "Mul" } mul
                && SoleConsumer(up.Output) == mul && mul.Output.Shape.Equals(gate.Output.Shape) && silu.Output.Shape.Equals(gate.Output.Shape))
            {
                return (gate, silu, mul);
            }
        }

        return null;
    }

    private void LowerBatchMatMul(Node n)
    {
        var a = n.Attributes;
        var x = RefF32(n.Inputs[0]);
        var z = RefF32(n.Inputs[1]);
        var y = NewDense(n.Output);
        AddStep(new BatchMatMulOp(x, z, y, a.Get("trans_a", false), a.Get("trans_b", false), a.Get("alpha", 1f)), x, z, y);
        _refs[n.Output] = y;
    }

    /// <summary>
    /// Another MatMulId of this segment not lowered yet that reads the same rows and ids as <paramref name="n"/> with a static weight of the same
    /// shape, type and precision switches (the gate and up projections of a layer's experts), or null. <c>SHARPLIGHT_CPU_NO_PAIR=1</c> turns it off.
    /// </summary>
    private Node? PairedMatMulId(Node n)
    {
        if (Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_NO_PAIR") is "1" or "true" || !_consumers.TryGetValue(n.Inputs[0], out var readers))
        {
            return null;
        }

        var weight = n.Inputs[1];
        foreach (var m in readers)
        {
            if (m != n && m.Op.Name == "MatMulId" && !_absorbed.Contains(m) && !_refs.ContainsKey(m.Output) && m.Inputs[2] == n.Inputs[2]
                && IsStatic(m.Inputs[1]) && m.Inputs[1].Type == weight.Type && m.Inputs[1].Shape.Equals(weight.Shape)
                && m.Attributes.Get(StandardOps.F32ActivationsAttribute, false) == n.Attributes.Get(StandardOps.F32ActivationsAttribute, false))
            {
                return m;
            }
        }

        return null;
    }
}
