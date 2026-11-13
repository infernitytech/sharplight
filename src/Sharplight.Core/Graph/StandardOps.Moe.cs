using Sharplight.Tensors;

namespace Sharplight.Graph;

// Mixture-of-experts building blocks (routing itself is Softmax + TopK + elementwise).
public static partial class StandardOps
{
    /// <summary>
    /// Expert-indexed matrix product (llama.cpp's mul_mat_id). x: [T, K] (every selected expert sees the token's row)
    /// or [T, S, K] (slot s of token t is multiplied, S = k); W: [E, N, K] (any float or quantized format, one [N, K]
    /// matrix per expert); ids: i32 [T, S] expert index per slot. Output [T, S, N] with
    /// y[t, s] = W[ids[t, s]] · x[t (, s)].
    /// </summary>
    public static readonly OpSchema MatMulId = new("MatMulId", OpCategory.MatMul, 3, 3, (i, a) =>
    {
        var x = i[0].Shape; var w = i[1].Shape; var ids = i[2].Shape;
        if (w.Rank != 3) throw new ArgumentException("MatMulId weight must be rank 3 [E, N, K].");
        if (ids.Rank != 2) throw new ArgumentException("MatMulId ids must be rank 2 [T, S].");
        if (x[-1] != w[2]) throw new ArgumentException($"MatMulId K mismatch: x{x} w{w}.");
        if (x.Rank == 3 && x[1] != ids[1]) throw new ArgumentException($"MatMulId slot mismatch: x{x} ids{ids}.");
        return [new TensorType(DataType.F32, new Shape(ids[0], ids[1], w[1]))];
    });
}
