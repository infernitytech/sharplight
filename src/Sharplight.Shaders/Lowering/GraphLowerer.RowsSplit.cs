using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Softmax and GroupNorm over long rows (rows_split.slang): rows are split into chunks across workgroups, a stats
/// pass writes per-chunk partials and an apply pass combines them, so a few long rows (a softmax over a whole
/// segmentation map, an instance norm over a large feature map) still fill the device. Shorter rows keep the
/// one-subgroup / one-workgroup-per-row kernels.
/// </summary>
public sealed partial class GraphLowerer
{
    /// <summary>Elements per chunk (16 per thread of a 256-thread workgroup).</summary>
    private const long SplitRowChunk = 4096;

    /// <summary>Shortest row the split kernels handle.</summary>
    private const long SplitRowMinLength = 4 * SplitRowChunk;

    /// <summary>Largest row count of the split kernels (rows run along the grid's y dimension).</summary>
    private const long SplitRowMaxRows = 65535;

    /// <summary>Lowers a softmax over dense rows of <paramref name="x"/> (see <see cref="RowsInput"/>) through the split kernels when its rows are long; false otherwise.</summary>
    private bool TryLowerSoftmaxSplit(Node n, TensorRef x, TensorRef y, long r1, long s0, long s1)
    {
        long d = x.Shape[^1];
        long rows = (x.Symbol is null ? x.Shape[0] : Bound(Dim.Sym(x.Symbol))) * r1;
        if (d < SplitRowMinLength || rows > SplitRowMaxRows)
        {
            return false;
        }

        long chunks = (d + SplitRowChunk - 1) / SplitRowChunk;
        var part = Temp(DataType.F32, [rows * chunks * 2], null);
        var macros = Macros(("D", d), ("R1", r1), ("S0", s0), ("S1", s1), ("SCALE", n.Attributes.Get("scale", 1f)),
            ("LOG_SOFTMAX", n.Attributes.Get("log", false)), ("CHUNK", SplitRowChunk));
        var tx = x;
        Func<Bindings, Grid> grid = b => new Grid((uint)chunks, (uint)(Rows(b, tx) * r1));
        Emit($"softmax {n.Output.Name} stats", [n], [Always(Lib("rows_split.slang", "softmax_stats", macros), grid)],
            [Pointer(x), Pointer(y), Pointer(part)], [x], [part]);
        Emit($"softmax {n.Output.Name}", [n], [Always(Lib("rows_split.slang", "softmax_apply", macros), grid)],
            [Pointer(x), Pointer(y), Pointer(part)], [x, part], [y]);
        return true;
    }

    /// <summary>Lowers a GroupNorm of dense <paramref name="x"/> [N, C, S...] through the split kernels when its groups are large; false otherwise.</summary>
    private bool TryLowerGroupNormSplit(Node n, TensorRef x, TensorRef? scale, TensorRef? bias, TensorRef y, long C, long S, long G)
    {
        long e = C / G * S;
        long rows = (x.Symbol is null ? x.Shape[0] : Bound(Dim.Sym(x.Symbol))) * G;
        if (e < SplitRowMinLength || rows > SplitRowMaxRows)
        {
            return false;
        }

        long chunks = (e + SplitRowChunk - 1) / SplitRowChunk;
        var part = Temp(DataType.F32, [rows * chunks * 2], null);
        var macros = Macros(("C", C), ("S", S), ("G", G), ("EPS", n.Attributes.Get("eps", 1e-5f)), ("HAS_SCALE", scale is not null),
            ("HAS_BIAS", bias is not null), ("CHUNK", SplitRowChunk));
        var tx = x;
        Func<Bindings, Grid> grid = b => new Grid((uint)chunks, (uint)(Rows(b, tx) * G));
        var sp = scale ?? x;
        var bp = bias ?? x;
        Emit($"groupnorm {n.Output.Name} stats", [n], [Always(Lib("rows_split.slang", "groupnorm_stats", macros), grid)],
            [Pointer(x), Pointer(sp), Pointer(bp), Pointer(y), Pointer(part)], [x], [part]);
        Emit($"groupnorm {n.Output.Name}", [n], [Always(Lib("rows_split.slang", "groupnorm_apply", macros), grid)],
            [Pointer(x), Pointer(sp), Pointer(bp), Pointer(y), Pointer(part)],
            [.. new[] { x, scale, bias, part }.Where(t => t is not null).Select(t => t!)], [y]);
        return true;
    }
}
