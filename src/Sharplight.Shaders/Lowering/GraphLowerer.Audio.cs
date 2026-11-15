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
        string weight = ToFloat(w.Type, $"p.in1[(o * {perGroup}u + c) * {taps}u + k]");
        string biasInit = bias is null ? "0.0" : "p.in2[o]";
        string body = $$"""
              uint t = d0, o = d1;
              uint g = o / {{outPerGroup}}u;
              float acc = {{biasInit}};
              for (uint k = 0u; k < {{taps}}u; k++) {
                int ti = int(t) * {{stride}} - {{padLeft}} + int(k) * {{dilation}};
                if (ti < 0 || ti >= int(p.tin)) { continue; }
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
}
