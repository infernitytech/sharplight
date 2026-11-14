using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

// Deformable convolution.
public sealed partial class GraphLowerer
{
    /// <summary>
    /// DeformConv (see <c>StandardOps.DeformConv</c>): one thread per output element sums, over the input channels of its group and
    /// the kernel taps, the weight times the bilinear sample of the input at the tap's regular position plus its offset (zero
    /// outside the image), scaled by the mask when there is one, then adds the bias.
    /// </summary>
    private void LowerDeformConv(Node n)
    {
        var a = n.Attributes;
        var x = AsDenseF32(Ref(n.Inputs[0]), "deform-x");
        var w = AsDenseF32(Ref(n.Inputs[1]), "deform-w");
        var offset = AsDenseF32(Ref(n.Inputs[2]), "deform-offset");
        var bias = AsDenseF32(Ref(n.Inputs[3]), "deform-bias");
        var mask = n.Inputs.Length > 4 ? AsDenseF32(Ref(n.Inputs[4]), "deform-mask") : null;
        var y = Produce(n.Output);
        var strides = a.Get("strides", new long[] { 1, 1 });
        var pads = a.Get("pads", new long[] { 0, 0, 0, 0 });
        var dilations = a.Get("dilations", new long[] { 1, 1 });
        long groups = a.Get("groups", 1L);
        long offsetGroups = a.Get("offset_groups", 1L);
        long c = x.Shape[1], h = x.Shape[2], wd = x.Shape[3];
        long o = w.Shape[0], kh = w.Shape[2], kw = w.Shape[3];
        long ho = y.Shape[2], wo = y.Shape[3];
        long points = ho * wo;
        long taps = kh * kw;
        long perGroup = c / groups;
        string body = $$"""
              uint b = d0, oc = d1, oh = d2, ow = d3;
              uint q = oh * {{wo}}u + ow;
              uint grp = oc / {{o / groups}}u;
              float acc = p.in3[oc];
              for (uint cl = 0; cl < {{perGroup}}u; cl++) {
                uint ch = grp * {{perGroup}}u + cl;
                uint og = ch / {{c / offsetGroups}}u;
                uint plane = (b * {{c}}u + ch) * {{h * wd}}u;
                for (uint tap = 0; tap < {{taps}}u; tap++) {
                  uint ob = ((b * {{offsetGroups}}u + og) * {{taps}}u + tap) * {{2 * points}}u + q;
                  float fy = float(int(oh * {{strides[0]}}u + (tap / {{kw}}u) * {{dilations[0]}}u) - {{pads[0]}}) + p.in2[ob];
                  float fx = float(int(ow * {{strides[1]}}u + (tap % {{kw}}u) * {{dilations[1]}}u) - {{pads[1]}}) + p.in2[ob + {{points}}u];
                  float y0 = floor(fy), x0 = floor(fx);
                  int yi = int(y0), xi = int(x0);
                  float ly = fy - y0, lx = fx - x0;
                  float v00 = 0.0, v01 = 0.0, v10 = 0.0, v11 = 0.0;
                  bool r0 = yi >= 0 && yi < {{h}}, r1 = yi + 1 >= 0 && yi + 1 < {{h}};
                  bool c0 = xi >= 0 && xi < {{wd}}, c1 = xi + 1 >= 0 && xi + 1 < {{wd}};
                  if (r0 && c0) { v00 = p.in0[plane + uint(yi * {{wd}} + xi)]; }
                  if (r0 && c1) { v01 = p.in0[plane + uint(yi * {{wd}} + xi + 1)]; }
                  if (r1 && c0) { v10 = p.in0[plane + uint((yi + 1) * {{wd}} + xi)]; }
                  if (r1 && c1) { v11 = p.in0[plane + uint((yi + 1) * {{wd}} + xi + 1)]; }
                  float s = (1.0 - ly) * ((1.0 - lx) * v00 + lx * v01) + ly * ((1.0 - lx) * v10 + lx * v11);
            {{(mask is null ? "" : $"      s *= p.in4[((b * {offsetGroups}u + og) * {taps}u + tap) * {points}u + q];")}}
                  acc += p.in1[(oc * {{perGroup}}u + cl) * {{taps}}u + tap] * s;
                }
              }
              p.out0[idx] = acc;
            """;
        EmitGen("deform_conv", [n], y.Shape, y.Symbol, mask is null ? [x, w, offset, bias] : [x, w, offset, bias, mask], [y], body);
    }
}
