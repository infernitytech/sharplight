using Sharplight.Graph;

namespace Sharplight.Shaders.Lowering;

// Discrete Fourier transforms too long for the importer's dense basis matmul.
public sealed partial class GraphLowerer
{
    /// <summary>Longest DFT on the device: the twiddle index k·n mod L is formed in 32-bit integers.</summary>
    private const long MaxDeviceDftLength = 1 << 16;

    /// <summary>
    /// Dft (see <c>StandardOps.Dft</c>) as a direct transform: one thread per (row, output bin) sums the L input terms with the twiddle
    /// angle 2π·(k·n mod L)/L (exact index arithmetic, f32 sums; the CPU provider runs an FFT in f64). The inverse of a one-sided
    /// spectrum (<c>real_output</c>) reads the Hermitian extension of the input bins on the fly, as numpy's irfft does.
    /// </summary>
    private void LowerDft(Node n)
    {
        var a = n.Attributes;
        var x = AsDenseF32(Ref(n.Inputs[0]), "dft");
        var y = Produce(n.Output);
        long length = a.Get<long>("length");
        long outBins = a.Get<long>("out_bins");
        bool inverse = a.Get("inverse", false);
        bool realOutput = a.Get("real_output", false);
        long inLength = x.Shape[^2];
        long inComponents = x.Shape[^1];
        long outComponents = realOutput ? 1 : 2;
        long used = Math.Min(inLength, realOutput ? (length / 2) + 1 : length);
        string sign = inverse ? "" : "-";
        string scale = inverse ? F(1f / length) : "1.0";
        string load = inComponents == 2
            ? "float re = p.in0[xb + m * 2u], im = p.in0[xb + m * 2u + 1u];"
            : "float re = p.in0[xb + m], im = 0.0;";
        string term = realOutput
            ? $$"""
                  // Bin m of the Hermitian extension: the input bins below `used` (DC and Nyquist real), their conjugates mirrored.
                  uint src = m; bool conj = false;
                  if (m >= {{used}}u) { src = {{length}}u - m; conj = true; if (src == 0u || src >= {{used}}u || m <= src) { continue; } }
                  uint xm = src;
                  float re = p.in0[xb + xm * {{inComponents}}u];
                  float im = {{(inComponents == 2 ? $"p.in0[xb + xm * 2u + 1u]" : "0.0")}};
                  if (xm == 0u{{(length % 2 == 0 ? $" || xm == {length / 2}u" : "")}}) { im = 0.0; }
                  if (conj) { im = -im; }
            """
            : $$"""
                  if (m >= {{used}}u) { break; }
                  {{load}}
            """;
        string body = $$"""
              uint row = idx / {{outBins}}u, k = idx % {{outBins}}u;
              uint xb = row * {{inLength * inComponents}}u;
              float sr = 0.0, si = 0.0;
              for (uint m = 0u; m < {{length}}u; m++) {
            {{term}}
                float angle = {{sign}}6.283185307179586 * float((k * m) % {{length}}u) / {{length}}.0;
                float c = cos(angle), s = sin(angle);
                sr += re * c - im * s;
                si += re * s + im * c;
              }
              uint ob = idx * {{outComponents}}u;
              p.out0[ob] = sr * {{scale}};
            {{(realOutput ? "" : $"  p.out0[ob + 1u] = si * {scale};")}}
            """;
        EmitGen("dft", [n], y.Shape[..^1], y.Rank > 2 ? y.Symbol : null, [x], [y], body);
    }
}
