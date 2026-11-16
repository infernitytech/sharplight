using Sharplight.Tensors;

namespace Sharplight.Graph;

// Sequence convolutions over a (possibly symbolic) time axis: the building blocks of audio codecs, vocoders and speech encoders.
public static partial class StandardOps
{
    /// <summary>
    /// 1-D convolution (cross-correlation, as in PyTorch's <c>Conv1d</c>) over a time-major signal: x f32 <c>[T, C]</c> (T may be symbolic),
    /// weight <c>[O, C/groups, K]</c> (any dense float type), optional bias <c>[O]</c>. Attributes: <c>stride</c> (default 1),
    /// <c>pad_left</c> and <c>pad_right</c> (zeros, default 0), <c>dilation</c> (default 1), <c>groups</c> (default 1; <c>groups</c> = C with
    /// one input channel per group is a depthwise convolution), <c>pad_mode</c> ("zeros" (default) or "reflect": the padding mirrors the signal
    /// without repeating the edge sample, as PyTorch's <c>padding_mode='reflect'</c>; the padding must be smaller than T). Output <c>[T_out, O]</c> with
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
    /// 1-D transposed convolution (PyTorch's <c>ConvTranspose1d</c>) over a time-major signal: x f32 <c>[T, C]</c> (T may be symbolic), weight
    /// <c>[C, O/groups, K]</c> (any dense float type), optional bias <c>[O]</c>. Every input position t adds <c>x[t] · w</c> at the output positions
    /// <c>t·stride - crop_left + k</c>. Attributes: <c>stride</c> (default 1), <c>crop_left</c> and <c>crop_right</c> (output positions dropped at either
    /// end, PyTorch's <c>padding</c>; default 0), <c>output_padding</c> (extra positions at the end, default 0), <c>groups</c> (default 1).
    /// Output <c>[(T - 1)·stride - crop_left - crop_right + K + output_padding, O]</c>.
    /// </summary>
    public static readonly OpSchema ConvTranspose1d = new("ConvTranspose1d", OpCategory.MatMul, 2, 3, (i, a) =>
    {
        var x = i[0].Shape;
        var w = i[1].Shape;
        if (x.Rank != 2 || w.Rank != 3)
        {
            throw new ArgumentException($"ConvTranspose1d expects x [T, C] and weight [C, O/groups, K], got x{x} w{w}.");
        }

        long groups = a.Get("groups", 1L);
        if (x[1].Fixed != w[0].Fixed || x[1].Fixed % groups != 0)
        {
            throw new ArgumentException($"ConvTranspose1d channel mismatch: x{x} w{w} groups={groups}.");
        }

        return [new(DataType.F32, new Shape(ConvTranspose1dLength(x[0], w[2].Fixed, a), w[1].Fixed * groups))];
    });

    /// <summary>
    /// Resamples a time-major signal <c>[T, C]</c> (T may be symbolic) by the rational factor <c>up / down</c> along time, as PyTorch's
    /// <c>F.interpolate</c> over one axis. Attributes: <c>mode</c> ("nearest" (source index <c>floor(j·down/up)</c>) or "linear" (half-pixel source
    /// position <c>(j + 0.5)·down/up - 0.5</c>, clamped to the signal; no anti-aliasing)), <c>up</c> and <c>down</c> (default 1). Output
    /// <c>[T·up/down (rounded down), C]</c>.
    /// </summary>
    public static readonly OpSchema Interpolate1d = new("Interpolate1d", OpCategory.Data, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        if (x.Rank != 2)
        {
            throw new ArgumentException($"Interpolate1d expects x [T, C], got {x}.");
        }

        long up = a.Get("up", 1L);
        long down = a.Get("down", 1L);
        return [new(DataType.F32, new Shape((x[0] * up).FloorDiv(down), x[1]))];
    });

    /// <summary>
    /// The recurrence of one LSTM direction over a time-major signal (PyTorch gate order input, forget, cell, output; zero initial state):
    /// <c>gates</c> f32 <c>[T, 4H]</c> holds the input projection <c>x·W_ihᵀ + b</c> (computed with a MatMul), <c>w_hh</c> <c>[4H, H]</c> the recurrent
    /// weights. Output: the hidden states <c>[T, H]</c>. Attribute <c>reverse</c> (default false) runs from the last position to the first (the
    /// output stays in time order); a bidirectional layer is two ops and a concatenation.
    /// </summary>
    public static readonly OpSchema Lstm = new("Lstm", OpCategory.Composite, 2, 2, (i, a) =>
    {
        var g = i[0].Shape;
        var w = i[1].Shape;
        if (g.Rank != 2 || w.Rank != 2 || g[1].Fixed != w[0].Fixed || w[0].Fixed != 4 * w[1].Fixed)
        {
            throw new ArgumentException($"Lstm expects gates [T, 4H] and w_hh [4H, H], got gates{g} w_hh{w}.");
        }

        return [new(DataType.F32, new Shape(g[0], w[1]))];
    });

    /// <summary>The output length of a 1-D transposed convolution of an input of <paramref name="length"/> positions with a kernel of <paramref name="kernel"/> taps.</summary>
    /// <param name="length">The input length (fixed or symbolic).</param>
    /// <param name="kernel">The number of taps.</param>
    /// <param name="a">The attributes <c>stride</c>, <c>crop_left</c>, <c>crop_right</c> and <c>output_padding</c>.</param>
    internal static Dim ConvTranspose1dLength(Dim length, long kernel, Attributes a)
    {
        long stride = a.Get("stride", 1L);
        long extra = kernel - stride - a.Get("crop_left", 0L) - a.Get("crop_right", 0L) + a.Get("output_padding", 0L);
        var scaled = length * stride;
        return extra >= 0 ? scaled + extra : scaled - (-extra);
    }

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
