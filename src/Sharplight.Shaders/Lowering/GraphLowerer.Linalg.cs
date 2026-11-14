using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

// Dense linear algebra beyond matrix products: Det and Inverse of small square matrices.
public sealed partial class GraphLowerer
{
    /// <summary>
    /// Det of x [..., M, M]: one thread per matrix runs an LU decomposition with partial pivoting (in f32; the CPU provider
    /// uses f64) on a dense copy of its matrix.
    /// </summary>
    private void LowerDet(Node n)
    {
        var x = Ref(n.Inputs[0]);
        long m = x.Shape[^1];
        var lu = Temp(DataType.F32, x.Shape, x.Symbol);
        EmitCopy(x, lu, "det:copy");
        var y = Produce(n.Output);
        string body = $$"""
              uint b0 = idx * {{m * m}}u;
              float det = 1.0;
              for (uint c = 0; c < {{m}}u; c++) {
                uint piv = c; float best = abs(p.out0[b0 + c * {{m}}u + c]);
                for (uint r = c + 1; r < {{m}}u; r++) { float v = abs(p.out0[b0 + r * {{m}}u + c]); if (v > best) { best = v; piv = r; } }
                if (best == 0.0) { det = 0.0; break; }
                if (piv != c) {
                  for (uint j = c; j < {{m}}u; j++) { float t = p.out0[b0 + c * {{m}}u + j]; p.out0[b0 + c * {{m}}u + j] = p.out0[b0 + piv * {{m}}u + j]; p.out0[b0 + piv * {{m}}u + j] = t; }
                  det = -det;
                }
                float d = p.out0[b0 + c * {{m}}u + c];
                det *= d;
                for (uint r = c + 1; r < {{m}}u; r++) {
                  float f = p.out0[b0 + r * {{m}}u + c] / d;
                  for (uint j = c + 1; j < {{m}}u; j++) { p.out0[b0 + r * {{m}}u + j] -= f * p.out0[b0 + c * {{m}}u + j]; }
                }
              }
              p.out1[idx] = det;
            """;
        EmitGen("det", [n], x.Shape[..^2], x.Symbol, [], [lu, y], body, outsAreReads: true);
    }

    /// <summary>
    /// Inverse of x [..., M, M]: one thread per matrix runs a Gauss-Jordan elimination with partial pivoting (in f32) on a
    /// dense copy of its matrix, turning the identity in the output into the inverse; a singular matrix yields non-finite values.
    /// </summary>
    private void LowerInverse(Node n)
    {
        var x = Ref(n.Inputs[0]);
        long m = x.Shape[^1];
        var a = Temp(DataType.F32, x.Shape, x.Symbol);
        EmitCopy(x, a, "inverse:copy");
        var y = Produce(n.Output);
        string body = $$"""
              uint b0 = idx * {{m * m}}u;
              for (uint r = 0; r < {{m}}u; r++) { for (uint j = 0; j < {{m}}u; j++) { p.out1[b0 + r * {{m}}u + j] = r == j ? 1.0 : 0.0; } }
              for (uint c = 0; c < {{m}}u; c++) {
                uint piv = c; float best = abs(p.out0[b0 + c * {{m}}u + c]);
                for (uint r = c + 1; r < {{m}}u; r++) { float v = abs(p.out0[b0 + r * {{m}}u + c]); if (v > best) { best = v; piv = r; } }
                if (piv != c) {
                  for (uint j = 0; j < {{m}}u; j++) {
                    float t = p.out0[b0 + c * {{m}}u + j]; p.out0[b0 + c * {{m}}u + j] = p.out0[b0 + piv * {{m}}u + j]; p.out0[b0 + piv * {{m}}u + j] = t;
                    float u = p.out1[b0 + c * {{m}}u + j]; p.out1[b0 + c * {{m}}u + j] = p.out1[b0 + piv * {{m}}u + j]; p.out1[b0 + piv * {{m}}u + j] = u;
                  }
                }
                float inv = 1.0 / p.out0[b0 + c * {{m}}u + c];
                for (uint j = 0; j < {{m}}u; j++) { p.out0[b0 + c * {{m}}u + j] *= inv; p.out1[b0 + c * {{m}}u + j] *= inv; }
                for (uint r = 0; r < {{m}}u; r++) {
                  if (r == c) { continue; }
                  float f = p.out0[b0 + r * {{m}}u + c];
                  for (uint j = 0; j < {{m}}u; j++) {
                    p.out0[b0 + r * {{m}}u + j] -= f * p.out0[b0 + c * {{m}}u + j];
                    p.out1[b0 + r * {{m}}u + j] -= f * p.out1[b0 + c * {{m}}u + j];
                  }
                }
              }
            """;
        EmitGen("inverse", [n], x.Shape[..^2], x.Symbol, [], [a, y], body, outsAreReads: true);
    }
}
