using Sharplight.Graph;

namespace Sharplight.Architectures.SeedVr;

/// <summary>
/// Causal 3-D convolutions over frames-first activations [F, C, H, W] that are processed chunk by chunk along time within one graph. A
/// kernel of kt time taps (stored folded as [O, kt·C, kh, kw], time-tap-major) sees the kt input frames ending at each output frame: the
/// first chunk is preceded by kt − 1 copies of its first frame, later chunks by the last kt − stride frames the convolution saw before
/// (its cache, kept per convolution between chunks). Spatial borders are zero-padded. Large inputs are convolved in pieces (groups of
/// output frames, then bands of output rows with their halo) so the stacked frame windows of one piece stay within a byte budget.
/// </summary>
internal sealed class CausalConvolution
{
    /// <summary>The stacked input of one convolution piece stays below this many bytes.</summary>
    private const long PieceBytes = 256L << 20;

    private readonly GraphBuilder _b;
    private readonly Dictionary<string, Value> _caches = new(StringComparer.Ordinal);

    /// <summary>Creates the helper for one graph.</summary>
    /// <param name="b">The graph.</param>
    public CausalConvolution(GraphBuilder b) => _b = b;

    /// <summary>Convolves one chunk of frames.</summary>
    /// <param name="x">The chunk [n, C, H, W].</param>
    /// <param name="prefix">The convolution's weights <c>{prefix}.weight</c> [O, kt·C, kh, kw] and optional <c>{prefix}.bias</c>; also its cache key.</param>
    /// <param name="temporalStride">Stride along time.</param>
    /// <param name="spatialStride">Stride along rows and columns.</param>
    /// <param name="pads">Zero padding (top, left, bottom, right), or <see langword="null"/> for "same" padding of odd kernels.</param>
    /// <returns>The output frames [(n + kt − 1 − ...) / stride, O, H', W'].</returns>
    public Value Apply(Value x, string prefix, int temporalStride = 1, int spatialStride = 1, long[]? pads = null)
    {
        var weight = _b.Weight(prefix + ".weight");
        var bias = _b.TryWeight(prefix + ".bias");
        long channels = x.Shape[1].Fixed;
        int taps = (int)(weight.Shape[1].Fixed / channels);
        long kernel = weight.Shape[2].Fixed;
        var padding = pads ?? [kernel / 2, kernel / 2, kernel / 2, kernel / 2];
        var padded = x;
        if (taps > 1)
        {
            if (_caches.TryGetValue(prefix, out var cache))
            {
                padded = _b.Concat(0, cache, x);
            }
            else
            {
                var first = _b.Slice(x, 0, 0, 1);
                padded = _b.Concat(0, [.. Enumerable.Repeat(first, taps - 1), x]);
            }

            int keep = taps - temporalStride;
            long frames = padded.Shape[0].Fixed;
            if (keep > 0)
            {
                _caches[prefix] = _b.Contiguous(_b.Slice(padded, 0, frames - keep, keep));
            }
        }

        long paddedFrames = padded.Shape[0].Fixed;
        long outputs = (paddedFrames - taps) / temporalStride + 1;
        long height = x.Shape[2].Fixed;
        long width = x.Shape[3].Fixed;
        long outputRows = (height + padding[0] + padding[2] - kernel) / spatialStride + 1;
        long frameBytes = taps * channels * (height + padding[0] + padding[2]) * (width + padding[1] + padding[3]) * sizeof(float);
        long group = temporalStride == 1 ? Math.Clamp(PieceBytes / Math.Max(1, frameBytes), 1, outputs) : 1;
        long bands = Math.Clamp((frameBytes + PieceBytes - 1) / PieceBytes, 1, outputRows);
        long bandRows = (outputRows + bands - 1) / bands;
        var pieces = new List<Value>();
        for (long start = 0; start < outputs; start += group)
        {
            long count = Math.Min(group, outputs - start);
            var rowsOut = new List<Value>();
            for (long row = 0; row < outputRows; row += bandRows)
            {
                long rows = Math.Min(bandRows, outputRows - row);
                rowsOut.Add(Piece(padded, weight, bias, taps, temporalStride, start, count, row, rows, kernel, spatialStride, padding, height));
            }

            pieces.Add(rowsOut.Count == 1 ? rowsOut[0] : _b.Concat(2, [.. rowsOut]));
        }

        return pieces.Count == 1 ? pieces[0] : _b.Concat(0, [.. pieces]);
    }

    /// <summary>Output frames [start, start + count) and rows [row, row + rows) of one convolution.</summary>
    private Value Piece(Value padded, Value weight, Value? bias, int taps, int temporalStride, long start, long count, long row, long rows, long kernel,
        int spatialStride, long[] padding, long height)
    {
        long first = row * spatialStride - padding[0];
        long last = (row + rows - 1) * spatialStride - padding[0] + kernel;
        long top = Math.Max(0, -first);
        long bottom = Math.Max(0, last - height);
        long inputStart = Math.Max(0, first);
        long inputRows = Math.Min(height, last) - inputStart;
        var windows = new Value[taps];
        for (int j = 0; j < taps; j++)
        {
            var frames = _b.Slice(padded, 0, start * temporalStride + j, temporalStride == 1 ? count : 1);
            windows[j] = inputStart == 0 && inputRows == height ? frames : _b.Slice(frames, 2, inputStart, inputRows);
        }

        var input = taps == 1 ? windows[0] : _b.Concat(1, windows);
        var attributes = Attributes.From(("strides", new long[] { spatialStride, spatialStride }), ("pads", new[] { top, padding[1], bottom, padding[3] }));
        return _b.Emit1("Conv", bias is null ? [input, weight] : [input, weight, bias], attributes);
    }
}
