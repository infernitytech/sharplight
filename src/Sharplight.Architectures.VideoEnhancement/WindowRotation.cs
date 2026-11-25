using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures.VideoEnhancement;

/// <summary>
/// The rotary position embedding of SeedVR2's window attention for one window layout. Positions are local to each token's window; every axis
/// (frame, row, column) rotates its own block of interleaved pairs with its own frequency series (<c>rope_frequencies</c>), the rest of the
/// head is left as it is.
/// <list type="bullet">
/// <item>3B: integer positions; the video's frame positions start after the text (frame + L), the text tokens sit at (p, p, p). Runs as the
/// <c>Rope</c> op with per-pair frequency divisors that give each axis its series.</item>
/// <item>7B: the video only, at positions spread evenly over [−1, 1] across the window on each axis (one position: −1); the angles are
/// constant per token and rotate through cosine and sine tables.</item>
/// </list>
/// </summary>
internal sealed class WindowRotation
{
    private const float RopeBase = 10000f;

    private readonly GraphBuilder _b;
    private readonly int _pairsPerAxis;
    private readonly int _rotated;
    private readonly int _headDim;
    private readonly bool _textRope;
    private readonly Value? _videoPositions;
    private readonly Value? _textPositions;
    private readonly Value? _factors;
    private readonly Value? _cos;
    private readonly Value? _sin;

    /// <summary>Declares the constants of one layout.</summary>
    /// <param name="b">The graph.</param>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="frequencies">The frequency series of one axis.</param>
    /// <param name="windows">The layout's windows.</param>
    /// <param name="rows">Rows of the token grid.</param>
    /// <param name="columns">Columns of the token grid.</param>
    /// <param name="textRows">Text tokens.</param>
    /// <param name="name">Name prefix of the constants.</param>
    public WindowRotation(GraphBuilder b, SeedVrHyperParameters hp, float[] frequencies, IReadOnlyList<VideoWindow> windows, int rows, int columns, int textRows, string name)
    {
        _b = b;
        _pairsPerAxis = frequencies.Length;
        _rotated = 3 * 2 * _pairsPerAxis;
        _headDim = hp.HeadDim;
        _textRope = hp.TextRope;
        int tokens = windows.Sum(w => w.Size);
        if (_textRope)
        {
            var positions = new int[tokens * 3];
            foreach (var window in windows)
            {
                ForEachToken(window, rows, columns, (token, f, r, c) =>
                {
                    positions[token * 3] = textRows + f;
                    positions[token * 3 + 1] = r;
                    positions[token * 3 + 2] = c;
                });
            }

            var text = new int[textRows * 3];
            for (int t = 0; t < textRows; t++)
            {
                text[t * 3] = t;
                text[t * 3 + 1] = t;
                text[t * 3 + 2] = t;
            }

            _videoPositions = b.Constant(name + ".rope_positions", DataType.I32, new Shape(tokens, 3), MemoryMarshal.AsBytes(positions.AsSpan()).ToArray());
            _textPositions = b.Constant(name + ".text_rope_positions", DataType.I32, new Shape(textRows, 3), MemoryMarshal.AsBytes(text.AsSpan()).ToArray());
            var factors = new float[_rotated / 2];
            for (int i = 0; i < factors.Length; i++)
            {
                double op = Math.Pow(RopeBase, -2.0 * i / _rotated);
                factors[i] = (float)(op / frequencies[i % _pairsPerAxis]);
            }

            _factors = b.Constant(name + ".rope_frequency_factors", DataType.F32, new Shape(factors.Length), MemoryMarshal.AsBytes(factors.AsSpan()).ToArray());
            return;
        }

        int pairs = 3 * _pairsPerAxis;
        var cos = new float[tokens * pairs];
        var sin = new float[tokens * pairs];
        foreach (var window in windows)
        {
            ForEachToken(window, rows, columns, (token, f, r, c) =>
            {
                Span<double> local = [Spread(f, window.Frames), Spread(r, window.Rows), Spread(c, window.Columns)];
                for (int axis = 0; axis < 3; axis++)
                {
                    for (int j = 0; j < _pairsPerAxis; j++)
                    {
                        double angle = local[axis] * frequencies[j];
                        int at = token * pairs + axis * _pairsPerAxis + j;
                        cos[at] = (float)Math.Cos(angle);
                        sin[at] = (float)Math.Sin(angle);
                    }
                }
            });
        }

        _cos = b.Constant(name + ".rope_cos", DataType.F32, new Shape(tokens, 1, pairs, 1), MemoryMarshal.AsBytes(cos.AsSpan()).ToArray());
        _sin = b.Constant(name + ".rope_sin", DataType.F32, new Shape(tokens, 1, pairs, 1), MemoryMarshal.AsBytes(sin.AsSpan()).ToArray());
    }

    /// <summary>Rotates video heads [tokens, H, hd] (tokens in grid order).</summary>
    /// <param name="heads">The queries or keys.</param>
    public Value RotateVideo(Value heads)
    {
        if (_textRope)
        {
            return Rope(heads, _videoPositions!);
        }

        long tokens = heads.Shape[0].Fixed;
        long headCount = heads.Shape[1].Fixed;
        int pairs = _rotated / 2;
        var rotated = _b.Reshape(_b.Contiguous(_b.Slice(heads, 2, 0, _rotated)), tokens, headCount, pairs, 2);
        var even = _b.Slice(rotated, 3, 0, 1);
        var odd = _b.Slice(rotated, 3, 1, 1);
        var newEven = _b.Sub(_b.Mul(even, _cos!), _b.Mul(odd, _sin!));
        var newOdd = _b.Add(_b.Mul(odd, _cos!), _b.Mul(even, _sin!));
        var joined = _b.Reshape(_b.Concat(3, newEven, newOdd), tokens, headCount, _rotated);
        return _rotated == _headDim ? joined : _b.Concat(2, joined, _b.Slice(heads, 2, _rotated, _headDim - _rotated));
    }

    /// <summary>Rotates text heads [L, H, hd] (3B), or returns them unchanged (7B).</summary>
    /// <param name="heads">The queries or keys.</param>
    public Value RotateText(Value heads) => _textRope ? Rope(heads, _textPositions!) : heads;

    private Value Rope(Value heads, Value positions)
        => _b.Emit1("Rope", [heads, positions, _factors!], Attributes.From(("dims", _rotated), ("base", RopeBase), ("style", "gptj"),
            ("sections", new[] { _pairsPerAxis, _pairsPerAxis, _pairsPerAxis, 0 })));

    /// <summary>torch.linspace(−1, 1, count)[index].</summary>
    private static double Spread(int index, int count) => count == 1 ? -1.0 : -1.0 + 2.0 * index / (count - 1);

    private static void ForEachToken(VideoWindow window, int rows, int columns, Action<int, int, int, int> visit)
    {
        for (int f = window.FrameStart; f < window.FrameEnd; f++)
        {
            for (int r = window.RowStart; r < window.RowEnd; r++)
            {
                for (int c = window.ColumnStart; c < window.ColumnEnd; c++)
                {
                    visit((f * rows + r) * columns + c, f - window.FrameStart, r - window.RowStart, c - window.ColumnStart);
                }
            }
        }
    }
}
