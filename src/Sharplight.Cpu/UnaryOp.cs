using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Unary elementwise op over f32 (activations, math functions, Scale and Clip), parallelized over rows or flat chunks.</summary>
internal sealed unsafe class UnaryOp : CpuOp
{
    private readonly UnaryKind _kind;
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly float _scale;
    private readonly float _bias;
    private readonly Action<int> _body;
    private int _chunks;
    private long _perChunk;
    private long _total;
    private long _cols;
    public override string Name => "Unary." + _kind;
    public UnaryOp(UnaryKind kind, TRef x, TRef y, float scale = 1, float bias = 0)
    {
        _kind = kind;
        _x = x;
        _y = y;
        _scale = scale;
        _bias = bias;
        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        _total = _x.Elements(p);
        if (_total == 0)
        {
            return;
        }

        _cols = _x.Cols(p);
        if (_x.IsDense)
        {
            _chunks = CpuParallel.Chunks(_total, 32768);
            _perChunk = (_total + _chunks - 1) / _chunks;
        }
        else
        {
            long rows = _x.Rows(p);
            _chunks = CpuParallel.Chunks(rows, Math.Max(1, 32768 / Math.Max(1, _cols)));
            _perChunk = (rows + _chunks - 1) / _chunks;
        }
        CpuParallel.For(_chunks, _body);
    }

    private void Body(int c)
    {
        if (_x.IsDense)
        {
            long s = c * _perChunk;
            long e = Math.Min(_total, s + _perChunk);
            if (s < e)
            {
                ScalarMath.Unary(_kind, (float*)_x.P + s, (float*)_y.P + s, e - s, _scale, _bias);
            }
        }
        else
        {
            long rows = _total / _cols;
            long s = c * _perChunk;
            long e = Math.Min(rows, s + _perChunk);
            for (long r = s; r < e; r++)
            {
                ScalarMath.Unary(_kind, _x.F(r), (float*)_y.P + r * _cols, _cols, _scale, _bias);
            }
        }
    }
}
