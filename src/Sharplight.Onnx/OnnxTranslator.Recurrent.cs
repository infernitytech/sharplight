using Sharplight.Graph;

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

    /// <summary>Weights of one direction. The recurrent biases may be null; <c>Recurrent</c> is also available as a constant for row slicing.</summary>
    private readonly record struct RnnWeights(Value Input, Value Recurrent, OnnxConst RecurrentConst, Value? InputBias, Value? RecurrentBias, string Name, int Direction);

    private static OnnxConst Slice0(OnnxConst c, long i)
    {
        long perItem = c.Count / c.Dims[0];
        int elementSize = OnnxType.ElementSize(c.ElemType);
        return OnnxConst.Lazy(c.ElemType, c.Dims[1..], () => c.Bytes().AsSpan((int)(i * perItem * elementSize), (int)(perItem * elementSize)).ToArray(), c.Name + $"[{i}]");
    }

    private static void ValidateRnnSupport(OnnxNode n, TranslatedTensor?[] ins)
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

        if (kind == "LSTM" && ins.Length > 7 && ins[7] is not null)
        {
            throw new NotSupportedException("LSTM peepholes");
        }

        if (ins.Length > 4 && ins[4] is not null)
        {
            throw new NotSupportedException("RNN sequence_lens");
        }
    }

    /// <summary>LSTM / GRU / RNN, unrolled over the (fixed) sequence length with one MatMul pair per step.</summary>
    private void Rnn(OnnxNode n, TranslatedTensor?[] ins)
    {
        ValidateRnnSupport(n, ins);
        var x = X(ins, 0);
        var w = ConstIn(ins, 1, "W");
        var r = ConstIn(ins, 2, "R");
        var bias = OptConst(ins, 3, "B");
        int layout = (int)n.Int("layout", 0);
        long hidden = n.Int("hidden_size", 0);
        if (hidden == 0)
        {
            hidden = r.Dims[2];
        }

        int seq = (int)(layout == 0 ? x.Shape[0].Fixed : x.Shape[1].Fixed);   // a symbolic sequence axis is rejected here
        long batch = layout == 0 ? x.Shape[1].Fixed : 0;
        var spec = new RnnSpec(n.OpType, layout, hidden, batch, seq, (int)w.Dims[0], n.Int("linear_before_reset", 0) != 0);
        string direction = n.Str("direction", "forward");

        var outputs = new Value?[spec.Seq][];   // [time][direction]
        var lastH = new Value[spec.Directions];
        var lastC = new Value?[spec.Directions];
        for (int d = 0; d < spec.Directions; d++)
        {
            bool reverse = direction == "reverse" || (direction == "bidirectional" && d == 1);
            var weights = LoadRnnWeights(Nm(n), w, r, bias, d, spec);
            var h = RnnInitialState(Opt(ins, 5), d, spec);
            var c = spec.Kind == "LSTM" ? RnnInitialState(Opt(ins, 6), d, spec) : null;
            for (int step = 0; step < spec.Seq; step++)
            {
                int t = reverse ? spec.Seq - 1 - step : step;
                (h, c) = RnnCell(spec, weights, RnnInput(x, t, spec), h, c);
                (outputs[t] ??= new Value?[spec.Directions])[d] = h;
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

        return new RnnWeights(input, recurrent, recurrentConst, inputBias, recurrentBias, name, d);
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
        var xw = Linear(xt, weights.Input, weights.InputBias);
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

    private Value Linear(Value input, Value weight, Value? bias)
    {
        var product = _b.MatMul(input, weight);
        Heavy();
        return bias is null ? product : _b.Add(product, bias);
    }

    /// <summary>x·W + b_W + h·Rᵀ + b_R, with the recurrent part omitted for the (zero) initial state.</summary>
    private Value RecurrentPreActivation(Value xw, Value? h, RnnWeights weights)
    {
        if (h is not null)
        {
            return _b.Add(xw, Linear(h, weights.Recurrent, weights.RecurrentBias));
        }

        return weights.RecurrentBias is null ? xw : _b.Add(xw, weights.RecurrentBias);
    }

    /// <summary>Gate <paramref name="gate"/> of a [batch, gates*H] tensor.</summary>
    private Value GatePart(Value v, int gate, long hidden) => _b.Slice(v, 1, gate * hidden, hidden);

    private (Value H, Value? C) LstmCell(RnnSpec spec, RnnWeights weights, Value xw, Value? h, Value? c)
    {
        var gates = RecurrentPreActivation(xw, h, weights);
        var input = _b.Sigmoid(GatePart(gates, 0, spec.Hidden));
        var output = _b.Sigmoid(GatePart(gates, 1, spec.Hidden));
        var forget = _b.Sigmoid(GatePart(gates, 2, spec.Hidden));
        var candidate = _b.Unary("Tanh", GatePart(gates, 3, spec.Hidden));
        var cell = c is null ? _b.Mul(input, candidate) : _b.Add(_b.Mul(forget, c), _b.Mul(input, candidate));
        return (_b.Mul(output, _b.Unary("Tanh", cell)), cell);
    }

    private Value GruCell(RnnSpec spec, RnnWeights weights, Value xw, Value? h)
    {
        long hidden = spec.Hidden;
        Value? hr = h is null ? null : Linear(h, weights.Recurrent, null);   // h·Rᵀ for the z, r and h gates

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
                candidate = _b.Add(candidate, Linear(resetState, RecurrentGateRows(weights, 2, hidden), null));
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
        var recurrent = weights.RecurrentConst;
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
