using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Recurrent networks (LSTM, GRU, RNN), unrolled over the sequence length.
internal sealed partial class OnnxTranslator
{
    /// <summary>Static properties of one LSTM/GRU/RNN node.</summary>
    /// <param name="Kind">"LSTM", "GRU" or "RNN".</param>
    /// <param name="Layout">0 = [seq, batch, ...], 1 = [batch, seq, ...].</param>
    /// <param name="Hidden">Hidden size.</param>
    /// <param name="Batch">Fixed batch size for layout 0; 0 (the symbolic batch) for layout 1.</param>
    /// <param name="Seq">Sequence length; the network is unrolled over it.</param>
    /// <param name="Directions">1 for forward/reverse, 2 for bidirectional.</param>
    /// <param name="LinearBeforeReset">GRU variant that applies the reset gate after the recurrent linear transform.</param>
    private readonly record struct RnnSpec(string Kind, int Layout, long Hidden, long Batch, int Seq, int Directions, bool LinearBeforeReset)
    {
        public int Gates => Kind == "LSTM" ? 4 : Kind == "GRU" ? 3 : 1;
    }

    /// <summary>
    /// Weights of one direction. The biases and LSTM peepholes (p_i, p_o, p_f) may be null. Constant weights are IR weights
    /// ([out, in] for MatMul) with <c>RecurrentConst</c> available for row slicing; runtime weights (<c>RecurrentConst</c> null)
    /// are [out, in] values multiplied with a transposed BatchMatMul.
    /// </summary>
    private readonly record struct RnnWeights(Value Input, Value Recurrent, OnnxConst? RecurrentConst, Value? InputBias, Value? RecurrentBias, Value? Peepholes,
        string Name, int Direction);

    private static OnnxConst Slice0(OnnxConst c, long i)
    {
        long perItem = c.Count / c.Dims[0];
        int elementSize = OnnxType.ElementSize(c.ElemType);
        return OnnxConst.Lazy(c.ElemType, c.Dims[1..], () => c.Bytes().AsSpan((int)(i * perItem * elementSize), (int)(perItem * elementSize)).ToArray(), c.Name + $"[{i}]");
    }

    private static void ValidateRnnSupport(OnnxNode n)
    {
        string kind = n.OpType;
        if (n.Has("activations") && n.Attributes["activations"].Strings.Count > 0)
        {
            var defaults = kind == "LSTM" ? new[] { "Sigmoid", "Tanh", "Tanh" } : kind == "GRU" ? new[] { "Sigmoid", "Tanh" } : new[] { "Tanh" };
            var requested = n.Attributes["activations"].Strings;
            if (!requested.Take(defaults.Length).SequenceEqual(defaults, StringComparer.OrdinalIgnoreCase) && requested.Count <= 2 * defaults.Length)
            {
                throw new NotSupportedException("custom RNN activations");
            }
        }

        if (n.Has("clip"))
        {
            throw new NotSupportedException("RNN cell clipping");
        }

    }

    /// <summary>LSTM / GRU / RNN, unrolled over the (fixed) sequence length with one MatMul pair per step.</summary>
    private void Rnn(OnnxNode n, TranslatedTensor?[] ins)
    {
        ValidateRnnSupport(n);
        var x = X(ins, 0);
        bool runtimeWeights = Enumerable.Range(1, 3).Any(i => ins.Length > i && ins[i] is { IsConst: false });
        var w = runtimeWeights ? null : ConstIn(ins, 1, "W");
        var r = runtimeWeights ? null : ConstIn(ins, 2, "R");
        var bias = runtimeWeights ? null : OptConst(ins, 3, "B");
        int layout = (int)n.Int("layout", 0);
        long hidden = n.Int("hidden_size", 0);
        long directions = runtimeWeights ? AsValue(ins[1]!).Shape[0].Fixed : w!.Dims[0];
        if (hidden == 0)
        {
            hidden = runtimeWeights ? AsValue(ins[2]!).Shape[2].Fixed : r!.Dims[2];
        }

        int seq = (int)(layout == 0 ? x.Shape[0].Fixed : x.Shape[1].Fixed);   // a symbolic sequence axis is rejected here
        long batch = layout == 0 ? x.Shape[1].Fixed : 0;
        var spec = new RnnSpec(n.OpType, layout, hidden, batch, seq, (int)directions, n.Int("linear_before_reset", 0) != 0);
        string direction = n.Str("direction", "forward");

        var outputs = new Value?[spec.Seq][];   // [time][direction]
        var lastH = new Value[spec.Directions];
        var lastC = new Value?[spec.Directions];
        for (int d = 0; d < spec.Directions; d++)
        {
            bool reverse = direction == "reverse" || (direction == "bidirectional" && d == 1);
            var weights = runtimeWeights ? RuntimeRnnWeights(Nm(n), ins, d, spec) : LoadRnnWeights(Nm(n), w!, r!, bias, d, spec);
            weights = weights with { Peepholes = RnnPeepholes(ins, d, spec) };
            var h = RnnInitialState(Opt(ins, 5), d, spec);
            var c = spec.Kind == "LSTM" ? RnnInitialState(Opt(ins, 6), d, spec) : null;
            var lengths = SequenceLengths(ins, spec, reverse);
            for (int step = 0; step < spec.Seq; step++)
            {
                int t = reverse ? spec.Seq - 1 - step : step;
                var (newH, newC) = RnnCell(spec, weights, RnnInput(x, t, spec), h, c);
                if (lengths is null)
                {
                    (h, c) = (newH, newC);
                    (outputs[t] ??= new Value?[spec.Directions])[d] = h;
                    continue;
                }

                // Past a batch entry's length the state is frozen and the output is zero.
                var active = _b.Binary("Greater", lengths, ConstValue(OnnxConst.Scalar(t), $"{Nm(n)}.t{t}"));
                var zero = ConstValue(OnnxConst.Scalar(0f), Nm(n) + ".zero");
                (outputs[t] ??= new Value?[spec.Directions])[d] = Emit("Where", [active, newH, zero]);
                h = Emit("Where", [active, newH, h ?? zero]);
                c = newC is null ? null : Emit("Where", [active, newC, c ?? zero]);
            }

            lastH[d] = h!;
            lastC[d] = c;
        }

        EmitRnnOutputs(n, spec, outputs, lastH, lastC);
    }

    private RnnWeights LoadRnnWeights(string name, OnnxConst w, OnnxConst r, OnnxConst? bias, int d, RnnSpec spec)
    {
        var input = WeightValue(Slice0(w, d), false, name + $".W{d}");
        var recurrentConst = Slice0(r, d);
        var recurrent = WeightValue(recurrentConst, false, name + $".R{d}");
        Value? inputBias = null;
        Value? recurrentBias = null;
        if (bias is not null)
        {
            // B is [Wb (gates*H), Rb (gates*H)] per direction.
            int width = (int)(spec.Gates * spec.Hidden);
            var values = Slice0(bias, d).F32();
            var wb = values.Take(width).Select(v => (double)v).ToArray();
            var rb = values.Skip(width).Take(width).Select(v => (double)v).ToArray();
            inputBias = ConstValue(OnnxConst.FromFloats(OnnxType.Float, [width], wb, name + $".bW{d}"), null);
            recurrentBias = ConstValue(OnnxConst.FromFloats(OnnxType.Float, [width], rb, name + $".bR{d}"), null);
        }

        return new RnnWeights(input, recurrent, recurrentConst, inputBias, recurrentBias, null, name, d);
    }

    /// <summary>Direction <paramref name="d"/> of runtime W [D, G·H, I], R [D, G·H, H] and B [D, 2·G·H].</summary>
    private RnnWeights RuntimeRnnWeights(string name, TranslatedTensor?[] ins, int d, RnnSpec spec)
    {
        long width = spec.Gates * spec.Hidden;
        Value Direction(int input)
        {
            var v = AsValue(ins[input]!);
            return Rs(_b.Slice(v, 0, d, 1), [.. v.Shape.Dims.Skip(1).Select(dim => dim.Fixed)]);
        }

        var input = Direction(1);
        var recurrent = Direction(2);
        Value? inputBias = null;
        Value? recurrentBias = null;
        if (ins.Length > 3 && ins[3] is not null)
        {
            var bias = Direction(3);
            inputBias = _b.Slice(bias, 0, 0, width);
            recurrentBias = _b.Slice(bias, 0, width, width);
        }

        return new RnnWeights(input, recurrent, null, inputBias, recurrentBias, null, name, d);
    }

    /// <summary>LSTM peephole weights [3·H] (p_i, p_o, p_f) of direction <paramref name="d"/>, or null.</summary>
    private Value? RnnPeepholes(TranslatedTensor?[] ins, int d, RnnSpec spec)
    {
        if (spec.Kind != "LSTM" || ins.Length <= 7 || ins[7] is null)
        {
            return null;
        }

        var p = ins[7]!;
        return p.IsConst ? ConstValue(Slice0(p.C!, d), null) : Rs(_b.Slice(p.V!, 0, d, 1), 3 * spec.Hidden);
    }

    /// <summary>Initial hidden/cell state of direction <paramref name="d"/>, or null when the input is absent (zeros).</summary>
    private Value? RnnInitialState(TranslatedTensor? initial, int d, RnnSpec spec)
    {
        if (initial is null)
        {
            return null;
        }

        if (initial.IsConst)
        {
            return ConstValue(Slice0(initial.C!, d), null);
        }

        return spec.Layout == 0
            ? Rs(_b.Slice(initial.V!, 0, d, 1), spec.Batch, spec.Hidden)
            : Rs(_b.Slice(initial.V!, 1, d, 1), 0, spec.Hidden);
    }

    /// <summary>
    /// sequence_lens as f32 [batch, 1], or null when absent or a constant covering the whole sequence. Reverse directions
    /// start at each entry's own last step, which the unrolled network does not express.
    /// </summary>
    private Value? SequenceLengths(TranslatedTensor?[] ins, RnnSpec spec, bool reverse)
    {
        if (ins.Length <= 4 || ins[4] is not { } lengths || (lengths.IsConst && lengths.C!.Ints().All(l => l == spec.Seq)))
        {
            return null;
        }

        if (reverse)
        {
            throw new NotSupportedException("RNN sequence_lens shorter than the sequence in a reverse direction");
        }

        var v = AsValue(lengths);
        var f = v.Type == DataType.F32 ? v : Emit("Cast", [v], Attributes.From(("type", "f32")));
        return spec.Layout == 0 ? Rs(f, spec.Batch, 1) : Rs(f, 0, 1);
    }

    /// <summary>The [batch, input] slice of time step <paramref name="t"/>.</summary>
    private Value RnnInput(Value x, int t, RnnSpec spec)
    {
        long inputSize = x.Shape[2].Fixed;
        return spec.Layout == 0
            ? Rs(_b.Slice(x, 0, t, 1), spec.Batch, inputSize)
            : Rs(_b.Slice(x, 1, t, 1), 0, inputSize);
    }

    /// <summary>One time step: returns the new hidden state and (LSTM only) cell state.</summary>
    private (Value H, Value? C) RnnCell(RnnSpec spec, RnnWeights weights, Value xt, Value? h, Value? c)
    {
        var xw = Linear(xt, weights.Input, weights.InputBias, weights.RecurrentConst is null);
        switch (spec.Kind)
        {
            case "RNN":
                return (_b.Unary("Tanh", RecurrentPreActivation(xw, h, weights)), c);
            case "LSTM":
                return LstmCell(spec, weights, xw, h, c);
            default:
                return (GruCell(spec, weights, xw, h), c);
        }
    }

    /// <summary>input · weightᵀ (+ bias) for an [out, in] IR weight or runtime value.</summary>
    private Value Linear(Value input, Value weight, Value? bias, bool runtimeWeight)
    {
        var product = runtimeWeight ? Emit("BatchMatMul", [input, weight], Attributes.From(("trans_b", true))) : _b.MatMul(input, weight);
        Heavy();
        return bias is null ? product : _b.Add(product, bias);
    }

    /// <summary>x·W + b_W + h·Rᵀ + b_R, with the recurrent part omitted for the (zero) initial state.</summary>
    private Value RecurrentPreActivation(Value xw, Value? h, RnnWeights weights)
    {
        if (h is not null)
        {
            return _b.Add(xw, Linear(h, weights.Recurrent, weights.RecurrentBias, weights.RecurrentConst is null));
        }

        return weights.RecurrentBias is null ? xw : _b.Add(xw, weights.RecurrentBias);
    }

    /// <summary>Gate <paramref name="gate"/> of a [batch, gates*H] tensor.</summary>
    private Value GatePart(Value v, int gate, long hidden) => _b.Slice(v, 1, gate * hidden, hidden);

    private (Value H, Value? C) LstmCell(RnnSpec spec, RnnWeights weights, Value xw, Value? h, Value? c)
    {
        var gates = RecurrentPreActivation(xw, h, weights);
        var peepholes = weights.Peepholes;
        Value Peep(Value preActivation, int which, Value? state) =>
            peepholes is null || state is null ? preActivation : _b.Add(preActivation, _b.Mul(_b.Slice(peepholes, 0, which * spec.Hidden, spec.Hidden), state));

        var input = _b.Sigmoid(Peep(GatePart(gates, 0, spec.Hidden), 0, c));
        var forget = _b.Sigmoid(Peep(GatePart(gates, 2, spec.Hidden), 2, c));
        var candidate = _b.Unary("Tanh", GatePart(gates, 3, spec.Hidden));
        var cell = c is null ? _b.Mul(input, candidate) : _b.Add(_b.Mul(forget, c), _b.Mul(input, candidate));
        var output = _b.Sigmoid(Peep(GatePart(gates, 1, spec.Hidden), 1, cell));
        return (_b.Mul(output, _b.Unary("Tanh", cell)), cell);
    }

    private Value GruCell(RnnSpec spec, RnnWeights weights, Value xw, Value? h)
    {
        long hidden = spec.Hidden;
        Value? hr = h is null ? null : Linear(h, weights.Recurrent, null, weights.RecurrentConst is null);   // h·Rᵀ for the z, r and h gates

        Value UpdateOrResetGate(int gate)
        {
            var v = GatePart(xw, gate, hidden);
            if (hr is not null)
            {
                v = _b.Add(v, GatePart(hr, gate, hidden));
            }

            if (weights.RecurrentBias is not null)
            {
                v = _b.Add(v, _b.Slice(weights.RecurrentBias, 0, gate * hidden, hidden));
            }

            return v;
        }

        var z = _b.Sigmoid(UpdateOrResetGate(0));
        var r = _b.Sigmoid(UpdateOrResetGate(1));
        var candidate = GatePart(xw, 2, hidden);
        if (spec.LinearBeforeReset)
        {
            var recurrent = h is null ? null : GatePart(hr!, 2, hidden);
            if (weights.RecurrentBias is not null)
            {
                var recurrentBias = _b.Slice(weights.RecurrentBias, 0, 2 * hidden, hidden);
                recurrent = recurrent is null ? recurrentBias : _b.Add(recurrent, recurrentBias);
            }

            if (recurrent is not null)
            {
                candidate = _b.Add(candidate, _b.Mul(r, recurrent));
            }
        }
        else
        {
            if (h is not null)
            {
                var resetState = _b.Mul(r, h);
                candidate = _b.Add(candidate, Linear(resetState, RecurrentGateRows(weights, 2, hidden), null, weights.RecurrentConst is null));
            }

            if (weights.RecurrentBias is not null)
            {
                candidate = _b.Add(candidate, _b.Slice(weights.RecurrentBias, 0, 2 * hidden, hidden));
            }
        }

        candidate = _b.Unary("Tanh", candidate);
        var keep = _b.Scale(z, -1f, 1f);   // 1 - z
        return h is null ? _b.Mul(keep, candidate) : _b.Add(_b.Mul(keep, candidate), _b.Mul(z, h));
    }

    /// <summary>Rows [g*H, (g+1)*H) of a recurrent weight [G*H, H] as a new weight.</summary>
    private Value RecurrentGateRows(RnnWeights weights, int gate, long hidden)
    {
        if (weights.RecurrentConst is not { } recurrent)
        {
            return _b.Slice(weights.Recurrent, 0, gate * hidden, hidden);
        }

        long cols = recurrent.Dims[1];
        long elementSize = OnnxType.ElementSize(recurrent.ElemType);
        string name = weights.Name + $".Rh{weights.Direction}";
        var rows = OnnxConst.Lazy(recurrent.ElemType, [hidden, cols], () => recurrent.Bytes().AsSpan((int)(gate * hidden * cols * elementSize), (int)(hidden * cols * elementSize)).ToArray(), name);
        return WeightValue(rows, false, name);
    }

    /// <summary>Y: layout 0 [seq, D, B, H], layout 1 [B, seq, D, H]; Y_h and Y_c are stacked over directions.</summary>
    private void EmitRnnOutputs(OnnxNode n, RnnSpec spec, Value?[][] outputs, Value[] lastH, Value?[] lastC)
    {
        bool layout0 = spec.Layout == 0;
        if (n.Outputs.Count > 0 && n.Outputs[0] != "")
        {
            var perTime = new List<Value>();
            for (int t = 0; t < spec.Seq; t++)
            {
                var states = outputs[t].Select(v => layout0 ? Rs(v!, 1, 1, spec.Batch, spec.Hidden) : Rs(v!, 0, 1, 1, spec.Hidden)).ToArray();
                perTime.Add(spec.Directions == 1 ? states[0] : _b.Concat(layout0 ? 1 : 2, states));
            }

            SetDyn(n, 0, perTime.Count == 1 ? perTime[0] : ConcatMany(layout0 ? 0 : 1, perTime.ToArray()));
        }

        if (n.Outputs.Count > 1 && n.Outputs[1] != "")
        {
            SetDyn(n, 1, StackDirections(spec, lastH));
        }

        if (n.Outputs.Count > 2 && n.Outputs[2] != "")
        {
            SetDyn(n, 2, StackDirections(spec, lastC.Select(c => c ?? throw new NotSupportedException("Y_c requested for a non-LSTM")).ToArray()));
        }
    }

    private Value StackDirections(RnnSpec spec, Value[] perDirection)
    {
        bool layout0 = spec.Layout == 0;
        Value Shaped(Value v) => layout0 ? Rs(v, 1, spec.Batch, spec.Hidden) : Rs(v, 0, 1, spec.Hidden);
        if (spec.Directions == 1)
        {
            return Shaped(perDirection[0]);
        }

        return ConcatMany(layout0 ? 0 : 1, perDirection.Select(Shaped).ToArray());
    }
}
