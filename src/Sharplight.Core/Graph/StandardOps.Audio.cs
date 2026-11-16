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

    /// <summary>
    /// 2-D convolution (cross-correlation) over a time-major feature map: x f32 <c>[T, C, W]</c> (T, the time axis, may be symbolic; W is a
    /// fixed feature axis such as mel bins), weight <c>[O, C/groups, KT, KW]</c> (any dense float type), optional bias <c>[O]</c>.
    /// Attributes: <c>strides</c> long[2] (time, feature; default 1, 1), <c>pads</c> long[4] (zeros: time before, feature before, time after,
    /// feature after; default 0), <c>groups</c> (default 1; C groups of one channel is a depthwise convolution). Output
    /// <c>[T_out, O, W_out]</c> with <c>T_out = (T + pads[0] + pads[2] − KT) / strides[0] + 1</c> and likewise W_out. The subsampling front
    /// end of conformer speech encoders (strided 3×3 convolutions over [frames, channels, mel bins]); the time axis stays outermost so its
    /// length can vary per run, which the image-layout <c>Conv</c> (fixed spatial axes) does not allow.
    /// </summary>
    public static readonly OpSchema SequenceConv2d = new("SequenceConv2d", OpCategory.MatMul, 2, 3, (i, a) =>
    {
        var x = i[0].Shape;
        var w = i[1].Shape;
        if (x.Rank != 3 || w.Rank != 4)
        {
            throw new ArgumentException($"SequenceConv2d expects x [T, C, W] and weight [O, C/groups, KT, KW], got x{x} w{w}.");
        }

        long groups = a.Get("groups", 1L);
        if (x[1].Fixed != w[1].Fixed * groups || w[0].Fixed % groups != 0)
        {
            throw new ArgumentException($"SequenceConv2d channel mismatch: x{x} w{w} groups={groups}.");
        }

        var strides = a.Get("strides", new long[] { 1, 1 });
        var pads = a.Get("pads", new long[] { 0, 0, 0, 0 });
        var time = Conv1dLength(x[0], w[2].Fixed, Attributes.From(("stride", strides[0]), ("pad_left", pads[0]), ("pad_right", pads[2])));
        long width = ((x[2].Fixed + pads[1] + pads[3] - w[3].Fixed) / strides[1]) + 1;
        return [new(DataType.F32, new Shape(time, w[0], width))];
    });

    /// <summary>
    /// Self-attention with Transformer-XL relative positions (conformer encoders: Parakeet, Canary): queries <c>qc</c> and <c>qp</c>
    /// f32 <c>[T, H, D]</c> (the query plus the content bias u and plus the position bias v), keys k <c>[T, H, D]</c>, values v
    /// <c>[T, H, Dv]</c> and the projected relative position embeddings p <c>[2T − 1, H, D]</c>, row r for the distance T − 1 − r (from
    /// T − 1 down to −(T − 1)). Query i attends every key j with the score
    /// <c>scale · (qc_i · k_j + qp_i · p_(T − 1 − i + j))</c>, the position term being that of the distance i − j. Attribute <c>scale</c>.
    /// Output <c>[T, H, Dv]</c>.
    /// </summary>
    public static readonly OpSchema RelativeAttention = new("RelativeAttention", OpCategory.Composite, 5, 5, (i, a) =>
    {
        var q = i[0].Shape;
        var v = i[3].Shape;
        if (q.Rank != 3 || i[1].Shape.Rank != 3 || i[2].Shape.Rank != 3 || v.Rank != 3 || i[4].Shape.Rank != 3)
        {
            throw new ArgumentException($"RelativeAttention expects rank-3 qc, qp, k, v and p, got {q}, {i[1].Shape}, {i[2].Shape}, {v}, {i[4].Shape}.");
        }

        return [new(DataType.F32, new Shape(q[0], q[1], v[2]))];
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
