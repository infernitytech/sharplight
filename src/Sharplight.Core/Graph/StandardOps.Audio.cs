using Sharplight.Tensors;

namespace Sharplight.Graph;

// Sequence convolutions over a (possibly symbolic) time axis: the building blocks of audio codecs, vocoders and speech encoders.
public static partial class StandardOps
{
    /// <summary>
    /// 1-D convolution (cross-correlation, as in PyTorch's <c>Conv1d</c>) over a time-major signal: x f32 <c>[T, C]</c> (T may be symbolic),
    /// weight <c>[O, C/groups, K]</c> (any dense float type), optional bias <c>[O]</c>. Attributes: <c>stride</c> (default 1),
    /// <c>pad_left</c> and <c>pad_right</c> (zeros, default 0), <c>dilation</c> (default 1), <c>groups</c> (default 1; <c>groups</c> = C with
    /// one input channel per group is a depthwise convolution). Output <c>[T_out, O]</c> with
    /// <c>T_out = (T + pad_left + pad_right - dilation·(K - 1) - 1) / stride + 1</c>.
    /// </summary>
    public static readonly OpSchema Conv1d = new("Conv1d", OpCategory.MatMul, 2, 3, (i, a) =>
    {
        var x = i[0].Shape;
        var w = i[1].Shape;
        if (x.Rank != 2 || w.Rank != 3)
        {
            throw new ArgumentException($"Conv1d expects x [T, C] and weight [O, C/groups, K], got x{x} w{w}.");
        }

        long groups = a.Get("groups", 1L);
        if (x[1].Fixed != w[1].Fixed * groups || w[0].Fixed % groups != 0)
        {
            throw new ArgumentException($"Conv1d channel mismatch: x{x} w{w} groups={groups}.");
        }

        return [new(DataType.F32, new Shape(Conv1dLength(x[0], w[2].Fixed, a), w[0]))];
    });

    /// <summary>The output length of a 1-D convolution of an input of <paramref name="length"/> positions with a kernel of <paramref name="kernel"/> taps.</summary>
    /// <param name="length">The input length (fixed or symbolic).</param>
    /// <param name="kernel">The number of taps.</param>
    /// <param name="a">The attributes <c>stride</c>, <c>pad_left</c>, <c>pad_right</c> and <c>dilation</c>.</param>
    internal static Dim Conv1dLength(Dim length, long kernel, Attributes a)
    {
        long stride = a.Get("stride", 1L);
        long span = (a.Get("dilation", 1L) * (kernel - 1)) + 1;
        long extra = a.Get("pad_left", 0L) + a.Get("pad_right", 0L) - span;
        var padded = extra >= 0 ? length + extra : length - (-extra);
        return (stride == 1 ? padded : padded.FloorDiv(stride)) + 1;
    }
}
