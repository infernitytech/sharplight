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
        var disk = DiskExperts(ins[1]);
        // A gate / up pair runs as one op when both weights are resident or both are on disk.
        if (IsStatic(ins[1]) && PairedMatMulId(n) is { } pair && (disk is null) == (DiskExperts(pair.Inputs[1]) is null))
        {
            // Gate and up projections of the experts: one pass over both, with the gated activation SiLU(gate) * up when it follows.
            var w2 = Ref(pair.Inputs[1]);
            var y2 = NewDense(pair.Output);
            var disk2 = DiskExperts(pair.Inputs[1]);
            if (GatedActivation(n, pair) is { } gated)
            {
                bool gateFirst = gated.Gate == n;
                var h = NewDense(gated.Mul.Output);
                var (gateWeight, gateOut, upWeight, upOut) = gateFirst ? (w, y, w2, y2) : (w2, y2, w, y);
                var binding = Binding(gateFirst ? disk : disk2, gateFirst ? disk2 : disk);
                AddStep(new MatMulIdOp(x, gateWeight, expertIndex, gateOut, format, intDot, upWeight, upOut, h, disk: binding), x, w, expertIndex, y, w2, y2, h);
                _refs[gated.Silu.Output] = gateOut;
                _refs[gated.Mul.Output] = h;
                _absorbed.Add(gated.Silu);
                _absorbed.Add(gated.Mul);
            }
            else
            {
                AddStep(new MatMulIdOp(x, w, expertIndex, y, format, intDot, w2, y2, disk: Binding(disk, disk2)), x, w, expertIndex, y, w2, y2);
            }

            _refs[n.Output] = y;
            _refs[pair.Output] = y2;
            _absorbed.Add(pair);
            return;
        }

        if (WeightedSlotSum(n) is { } sum && RefF32(sum.Weights) is { IsDense: true } slotWeights)
        {
            // The weighted sum over the slots that follows (an MoE layer's output): written by the product, y is its scratch.
            var summed = NewDense(sum.ReduceSum.Output);
            AddStep(new MatMulIdOp(x, w, expertIndex, y, format, intDot, slotWeights: slotWeights, slotSum: summed, disk: Binding(disk, null)), x, w, expertIndex, y, slotWeights, summed);
            _refs[n.Output] = y;
            _refs[sum.Mul.Output] = y;
            _refs[sum.Permute.Output] = y;
            _refs[sum.ReduceSum.Output] = summed;
            _absorbed.Add(sum.Mul);
            _absorbed.Add(sum.Permute);
            _absorbed.Add(sum.ReduceSum);
            return;
        }

        AddStep(new MatMulIdOp(x, w, expertIndex, y, format, intDot, disk: Binding(disk, null)), x, w, expertIndex, y);
        _refs[n.Output] = y;
    }

    /// <summary>The disk-backed form of an expert weight (<see cref="CompileOptions.DiskExperts"/>), or null when it is resident.</summary>
    private Runtime.Experts.DiskExpertTensor? DiskExperts(Value weight)
        => weight.Weight is { } w && _options.DiskExperts is { } store ? store.Find(w) : null;

    /// <summary>The expert cache binding of an op whose first (and second) weight are on disk, or null when they are resident.</summary>
    private DiskExpertBinding? Binding(Runtime.Experts.DiskExpertTensor? first, Runtime.Experts.DiskExpertTensor? second)
        => first is null ? null : new DiskExpertBinding(_options.DiskExperts!, first, second);

    /// <summary>
    /// The weighted slot sum after an expert product <paramref name="n"/> ([T, S, N]): <c>ReduceSum(Permute(Mul(y, w), 0, 2, 1))</c> with slot
    /// weights w of [T, S, 1], each node the sole consumer of the previous one; null when the product is not followed by it.
    /// </summary>
    private (Value Weights, Node Mul, Node Permute, Node ReduceSum)? WeightedSlotSum(Node n)
    {
        var shape = n.Output.Shape;
        if (shape.Rank != 3 || SoleConsumer(n.Output) is not { Op.Name: "Mul" } mul || !mul.Output.Shape.Equals(shape))
        {
            return null;
        }

        var weights = mul.Inputs[0] == n.Output ? mul.Inputs[1] : mul.Inputs[0];
        if (weights.Shape.Rank != 3 || !weights.Shape[0].Equals(shape[0]) || !weights.Shape[1].Equals(shape[1]) || weights.Shape[2] is not { IsSymbolic: false, Fixed: 1 }
            || !_refs.ContainsKey(weights) && !IsStatic(weights))
        {
            return null;
        }

        if (SoleConsumer(mul.Output) is not { Op.Name: "Permute" } permute || !permute.Attributes.Get("perm", Array.Empty<int>()).AsSpan().SequenceEqual([0, 2, 1])
            || SoleConsumer(permute.Output) is not { Op.Name: "ReduceSum" } reduce || reduce.Inputs.Length != 1
            || reduce.Output.Shape.Rank != 3 || !reduce.Output.Shape[1].Equals(shape[2]) || reduce.Output.Shape[2] is not { IsSymbolic: false, Fixed: 1 })
        {
            return null;
        }

        return (weights, mul, permute, reduce);
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
