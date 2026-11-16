using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

// Sequence convolutions over a time axis (audio codecs, vocoders, speech encoders).
public sealed partial class GraphLowerer
{
    /// <summary>
    /// Conv1d (see <c>StandardOps.Conv1d</c>) as a generated kernel: one thread per output (position, channel) sums the taps of its group's
    /// input channels with zero padding. The input length is read at run time, so the time axis may be symbolic. Weights of any float type
    /// are read in place.
    /// </summary>
    private void LowerConv1d(Node n)
    {
        var a = n.Attributes;
        var x = AsDenseF32(Ref(n.Inputs[0]), "conv1d");
        var w = EnsureDense(Ref(n.Inputs[1]), "conv1d-w");
        var bias = n.Inputs.Length > 2 ? AsDenseF32(Ref(n.Inputs[2]), "conv1d-bias") : null;
        var y = Produce(n.Output);
        long stride = a.Get("stride", 1L);
        long padLeft = a.Get("pad_left", 0L);
        long dilation = a.Get("dilation", 1L);
        long groups = a.Get("groups", 1L);
        long channels = x.Shape[1];
        long outputs = w.Shape[0];
        long perGroup = w.Shape[1];
        long taps = w.Shape[2];
        long outPerGroup = outputs / groups;
        bool reflect = a.Get("pad_mode", "zeros") == "reflect";
        string weight = ToFloat(w.Type, $"p.in1[(o * {perGroup}u + c) * {taps}u + k]");
        string edge = reflect
            ? "if (ti < 0) { ti = -ti; } else if (ti >= int(p.tin)) { ti = 2 * (int(p.tin) - 1) - ti; }"
            : "if (ti < 0 || ti >= int(p.tin)) { continue; }";
        string biasInit = bias is null ? "0.0" : "p.in2[o]";
        string body = $$"""
              uint t = d0, o = d1;
              uint g = o / {{outPerGroup}}u;
              float acc = {{biasInit}};
              for (uint k = 0u; k < {{taps}}u; k++) {
                int ti = int(t) * {{stride}} - {{padLeft}} + int(k) * {{dilation}};
                {{edge}}
                uint xb = uint(ti) * {{channels}}u + g * {{perGroup}}u;
                for (uint c = 0u; c < {{perGroup}}u; c++) {
                  acc += p.in0[xb + c] * {{weight}};
                }
              }
              p.out0[idx] = acc;
            """;
        KernelArg tin = x.Symbol is null ? new UIntArg((uint)x.Shape[0]) : new SymbolArg(x.Symbol, 1);
        TensorRef[] inputs = bias is null ? [x, w] : [x, w, bias];
        EmitGen($"conv1d {n.Output.Name}", [n], y.Shape, y.Symbol, inputs, [y], body, scalars: [("tin", tin)]);
    }

    /// <summary>
    /// ConvTranspose1d (see <c>StandardOps.ConvTranspose1d</c>) as a generated kernel: one thread per output (position, channel) gathers the input
    /// positions whose taps land on it.
    /// </summary>
    private void LowerConvTranspose1d(Node n)
    {
        var a = n.Attributes;
        var x = AsDenseF32(Ref(n.Inputs[0]), "convt1d");
        var w = EnsureDense(Ref(n.Inputs[1]), "convt1d-w");
        var bias = n.Inputs.Length > 2 ? AsDenseF32(Ref(n.Inputs[2]), "convt1d-bias") : null;
        var y = Produce(n.Output);
        long stride = a.Get("stride", 1L);
        long cropLeft = a.Get("crop_left", 0L);
        long groups = a.Get("groups", 1L);
        long channels = x.Shape[1];
        long perGroup = channels / groups;
        long outPerGroup = w.Shape[1];
        long taps = w.Shape[2];
        string weight = ToFloat(w.Type, $"p.in1[((g * {perGroup}u + c) * {outPerGroup}u + oi) * {taps}u + k]");
        string biasInit = bias is null ? "0.0" : "p.in2[o]";
        string body = $$"""
              uint u = d0, o = d1;
              uint g = o / {{outPerGroup}}u;
              uint oi = o % {{outPerGroup}}u;
              float acc = {{biasInit}};
              for (uint k = 0u; k < {{taps}}u; k++) {
                int num = int(u) + {{cropLeft}} - int(k);
                if (num < 0 || (num % {{stride}}) != 0) { continue; }
                int ti = num / {{stride}};
                if (ti >= int(p.tin)) { continue; }
                uint xb = uint(ti) * {{channels}}u + g * {{perGroup}}u;
                for (uint c = 0u; c < {{perGroup}}u; c++) {
                  acc += p.in0[xb + c] * {{weight}};
                }
              }
              p.out0[idx] = acc;
            """;
        KernelArg tin = x.Symbol is null ? new UIntArg((uint)x.Shape[0]) : new SymbolArg(x.Symbol, 1);
        TensorRef[] inputs = bias is null ? [x, w] : [x, w, bias];
        EmitGen($"convt1d {n.Output.Name}", [n], y.Shape, y.Symbol, inputs, [y], body, scalars: [("tin", tin)]);
    }

    /// <summary>Interpolate1d (see <c>StandardOps.Interpolate1d</c>) as a generated kernel: one thread per output (position, channel).</summary>
    private void LowerInterpolate1d(Node n)
    {
        var a = n.Attributes;
        var x = AsDenseF32(Ref(n.Inputs[0]), "interp1d");
        var y = Produce(n.Output);
        long up = a.Get("up", 1L);
        long down = a.Get("down", 1L);
        long channels = x.Shape[1];
        string body = a.Get("mode", "nearest") == "linear"
            ? $$"""
              uint j = d0, c = d1;
              int64_t num = int64_t(2 * j + 1) * {{down}} - {{up}};
              int i0 = 0;
              float frac = 0.0;
              if (num > 0) {
                int64_t den = 2 * {{up}};
                i0 = int(num / den);
                frac = float(num - int64_t(i0) * den) / float(den);
              }
              int i1 = min(i0 + 1, int(p.tin) - 1);
              i0 = min(i0, int(p.tin) - 1);
              float a0 = p.in0[uint(i0) * {{channels}}u + c];
              float a1 = p.in0[uint(i1) * {{channels}}u + c];
              p.out0[idx] = a0 + (a1 - a0) * frac;
            """
            : $$"""
              uint j = d0, c = d1;
              uint i = min(uint(((uint64_t(j) * {{down}}ul) / {{up}}ul)), p.tin - 1u);
              p.out0[idx] = p.in0[i * {{channels}}u + c];
            """;
        KernelArg tin = x.Symbol is null ? new UIntArg((uint)x.Shape[0]) : new SymbolArg(x.Symbol, 1);
        EmitGen($"interp1d {n.Output.Name}", [n], y.Shape, y.Symbol, [x], [y], body, scalars: [("tin", tin)]);
    }

    /// <summary>Lstm (see <c>StandardOps.Lstm</c>): one workgroup walks the time steps (<c>lstm.slang</c>).</summary>
    private void LowerLstm(Node n)
    {
        var gates = AsDenseF32(Ref(n.Inputs[0]), "lstm-gates");
        var w = AsDenseF32(Ref(n.Inputs[1]), "lstm-w");
        var y = Produce(n.Output);
        long h = w.Shape[1];
        var macros = Macros(("H", h), ("REVERSE", n.Attributes.Get("reverse", false)), ("PER", (h + 255) / 256));
        Emit($"lstm {n.Output.Name}", [n], [Always(Lib("lstm.slang", "lstm", macros), _ => new Grid(1))],
            [Pointer(gates), Pointer(w), Pointer(y), TokenArg(gates)], [gates, w], [y]);
    }
}
