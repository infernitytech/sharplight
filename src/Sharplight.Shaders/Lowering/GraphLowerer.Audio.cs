using System.Text;
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
    /// SequenceConv2d (see <c>StandardOps.SequenceConv2d</c>) as a generated kernel: one thread per output (time step, channel, feature
    /// position) sums the taps of its group's input channels with zero padding; the input length is read at run time.
    /// </summary>
    private void LowerSequenceConv2d(Node n)
    {
        var a = n.Attributes;
        var x = AsDenseF32(Ref(n.Inputs[0]), "seqconv2d");
        var w = EnsureDense(Ref(n.Inputs[1]), "seqconv2d-w");
        var bias = n.Inputs.Length > 2 ? AsDenseF32(Ref(n.Inputs[2]), "seqconv2d-bias") : null;
        var y = Produce(n.Output);
        var strides = a.Get("strides", new long[] { 1, 1 });
        var pads = a.Get("pads", new long[] { 0, 0, 0, 0 });
        long groups = a.Get("groups", 1L);
        long channels = x.Shape[1];
        long width = x.Shape[2];
        long outputs = w.Shape[0];
        long perGroup = w.Shape[1];
        long kt = w.Shape[2];
        long kw = w.Shape[3];
        long outPerGroup = outputs / groups;
        string weight = ToFloat(w.Type, $"p.in1[((o * {perGroup}u + c) * {kt}u + i) * {kw}u + j]");
        string biasInit = bias is null ? "0.0" : "p.in2[o]";
        string body = $$"""
              uint t = d0, o = d1, u = d2;
              uint g = o / {{outPerGroup}}u;
              float acc = {{biasInit}};
              for (uint i = 0u; i < {{kt}}u; i++) {
                int ti = int(t) * {{strides[0]}} - {{pads[0]}} + int(i);
                if (ti < 0 || ti >= int(p.tin)) { continue; }
                for (uint j = 0u; j < {{kw}}u; j++) {
                  int wi = int(u) * {{strides[1]}} - {{pads[1]}} + int(j);
                  if (wi < 0 || wi >= {{width}}) { continue; }
                  for (uint c = 0u; c < {{perGroup}}u; c++) {
                    acc += p.in0[(uint(ti) * {{channels}}u + g * {{perGroup}}u + c) * {{width}}u + uint(wi)] * {{weight}};
                  }
                }
              }
              p.out0[idx] = acc;
            """;
        KernelArg tin = x.Symbol is null ? new UIntArg((uint)x.Shape[0]) : new SymbolArg(x.Symbol, 1);
        TensorRef[] inputs = bias is null ? [x, w] : [x, w, bias];
        EmitGen($"seqconv2d {n.Output.Name}", [n], y.Shape, y.Symbol, inputs, [y], body, scalars: [("tin", tin)]);
    }

    /// <summary>
    /// RelativeAttention (see <c>StandardOps.RelativeAttention</c>): one workgroup per (query row, head). The scores are computed twice
    /// (once for the softmax maximum and sum, once per tile of keys whose weights the threads then apply to their output dimensions), so
    /// nothing scales with the row count but the loop trip counts.
    /// </summary>
    private void LowerRelativeAttention(Node n)
    {
        const int Threads = 64;
        var qc = AsDenseF32(Ref(n.Inputs[0]), "relattn-qc");
        var qp = AsDenseF32(Ref(n.Inputs[1]), "relattn-qp");
        var k = AsDenseF32(Ref(n.Inputs[2]), "relattn-k");
        var v = AsDenseF32(Ref(n.Inputs[3]), "relattn-v");
        var pos = AsDenseF32(Ref(n.Inputs[4]), "relattn-p");
        var spans = n.Inputs.Length > 5 ? DenseInput(n.Inputs[5], "relattn-spans") : null;
        var y = Produce(n.Output);
        long heads = qc.Shape[1];
        long dim = qc.Shape[2];
        long valueDim = v.Shape[2];
        long slots = (valueDim + Threads - 1) / Threads;
        float scale = n.Attributes.Get("scale", 1f);
        var sb = new StringBuilder();
        sb.AppendLine("#include \"math_ext.slang\"");
        sb.AppendLine(spans is null
            ? "struct Params { float* in0; float* in1; float* in2; float* in3; float* in4; float* out0; uint rows; };"
            : "struct Params { float* in0; float* in1; float* in2; float* in3; float* in4; int* in5; float* out0; uint rows; };");
        sb.AppendLine($"groupshared float red[{Threads}];");
        sb.AppendLine($"groupshared float red2[{Threads}];");
        sb.AppendLine($"groupshared float tile[{Threads}];");
        string range = spans is null
            ? "uint jFirst = 0u; uint jEnd = rows;"
            : "uint jFirst = uint(p.in5[i * 2u]); uint jEnd = min(rows, jFirst + uint(p.in5[i * 2u + 1u]));";
        sb.AppendLine($$"""
            float score(Params p, uint i, uint h, uint j) {
              uint rows = p.rows;
              uint q = (i * {{heads}}u + h) * {{dim}}u;
              uint kb = (j * {{heads}}u + h) * {{dim}}u;
              uint pb = ((rows - 1u - i + j) * {{heads}}u + h) * {{dim}}u;
              float s = 0.0;
              for (uint d = 0u; d < {{dim}}u; d++) {
                s += p.in0[q + d] * p.in2[kb + d] + p.in1[q + d] * p.in4[pb + d];
              }
              return s * {{F(scale)}};
            }
            [shader("compute")] [numthreads({{Threads}}, 1, 1)]
            void main(uint3 gid : SV_GroupID, uint3 lt : SV_GroupThreadID, uniform Params p) {
              uint rows = p.rows;
              uint i = gid.x / {{heads}}u;
              uint h = gid.x % {{heads}}u;
              if (i >= rows) return;
              uint t = lt.x;
              {{range}}
              float m = -3.0e38;
              float l = 0.0;
              for (uint j = jFirst + t; j < jEnd; j += {{Threads}}u) {
                float s = score(p, i, h, j);
                float mn = max(m, s);
                l = l * exp(m - mn) + exp(s - mn);
                m = mn;
              }
              red[t] = m;
              red2[t] = l;
              GroupMemoryBarrierWithGroupSync();
              for (uint w = {{Threads / 2}}u; w > 0u; w >>= 1) {
                if (t < w) {
                  float ma = red[t], mb = red[t + w];
                  float mn = max(ma, mb);
                  red2[t] = red2[t] * exp(ma - mn) + red2[t + w] * exp(mb - mn);
                  red[t] = mn;
                }
                GroupMemoryBarrierWithGroupSync();
              }
              float M = red[0];
              float L = red2[0];
              float acc[{{slots}}];
              for (uint r = 0u; r < {{slots}}u; r++) { acc[r] = 0.0; }
              for (uint j0 = jFirst; j0 < jEnd; j0 += {{Threads}}u) {
                uint j = j0 + t;
                tile[t] = j < jEnd ? exp(score(p, i, h, j) - M) : 0.0;
                GroupMemoryBarrierWithGroupSync();
                uint count = min({{Threads}}u, jEnd - j0);
                for (uint jj = 0u; jj < count; jj++) {
                  float wgt = tile[jj];
                  uint vb = ((j0 + jj) * {{heads}}u + h) * {{valueDim}}u;
                  for (uint r = 0u; r < {{slots}}u; r++) {
                    uint d = t + r * {{Threads}}u;
                    if (d < {{valueDim}}u) { acc[r] += wgt * p.in3[vb + d]; }
                  }
                }
                GroupMemoryBarrierWithGroupSync();
              }
              uint ob = (i * {{heads}}u + h) * {{valueDim}}u;
              for (uint r = 0u; r < {{slots}}u; r++) {
                uint d = t + r * {{Threads}}u;
                if (d < {{valueDim}}u) { p.out0[ob + d] = acc[r] / L; }
              }
            }
            """);
        var kernel = _compiler.Compile("relattn.slang", sb.ToString(), "main", _opt.Target, Macros(), _opt.Cache);
        KernelArg rows = qc.Symbol is null ? new UIntArg((uint)qc.Shape[0]) : new SymbolArg(qc.Symbol, 1);
        string? symbol = qc.Symbol;
        long fixedRows = qc.Shape[0];
        Emit($"relattn {n.Output.Name}", [n],
            [Always(kernel, b => new Grid((uint)((symbol is null ? fixedRows : b[symbol]) * heads)))],
            [Pointer(qc), Pointer(qp), Pointer(k), Pointer(v), Pointer(pos), .. spans is null ? [] : (KernelArg[])[Pointer(spans)], Pointer(y), rows],
            spans is null ? [qc, qp, k, v, pos] : [qc, qp, k, v, pos, spans], [y]);
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
