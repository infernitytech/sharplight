using Sharplight.Tensors;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Writes x [T, ...] into rows [offset, offset+T) of a state [ctx, ...] (f32 → state type: f32, f16, bf16, or q8_0 / q4_0 blocks).</summary>
internal sealed unsafe class CacheWriteOp : CpuOp
{
    private readonly TRef _state;
    private readonly TRef _x;
    private readonly string _offset;
    private readonly long _inner;
    private readonly long _rowLen;
    private readonly Action<int> _body;
    private readonly TRef? _seqRows;
    private readonly long _slotCtx;
    private readonly KvPages? _pages;   // paged cache: rows through the block table
    private readonly long _rowsPerToken;
    private readonly WeightFormat? _quantized;   // q8_0 / q4_0 cache: rows are quantized in 32-value blocks
    private long _past;
    private long _rows;
    private long _per;
    public override string Name => "CacheWrite";

    public CacheWriteOp(TRef state, TRef x, string offset, TRef? seqRows = null, int slots = 1, KvPages? pages = null)
    {
        _pages = pages;
        _state = state;
        _x = x;
        _offset = offset;
        _seqRows = seqRows;
        _slotCtx = state.Dims[0] / Math.Max(1, slots);
        _rowsPerToken = 1;
        for (int i = 1; i < x.Rank - 1; i++)
        {
            _rowsPerToken *= x.Dims[i];
        }

        _inner = 1;
        for (int i = 1; i < state.Rank; i++)
        {
            _inner *= state.Dims[i];
        }

        _rowLen = x.Dims[^1];
        if (Quant.FormatOf(state.Type) is { } format && Quant.CanQuantize(format))
        {
            _quantized = format;
            if (_rowLen % 32 != 0)
            {
                throw new ArgumentException($"CacheWrite: {state.Type} cache rows must be a multiple of 32 values.");
            }
        }

        long xi = 1;
        for (int i = 1; i < x.Rank; i++)
        {
            xi *= x.Dims[i];
        }

        if (xi != _inner)
        {
            throw new ArgumentException("CacheWrite: x inner size does not match state inner size.");
        }

        _body = Body;
    }

    public override void Run(RunParameters p)
    {
        long t = _x.Dim(0, p);
        if (t == 0)
        {
            return;
        }

        if (_seqRows is not null)
        {
            int* rw = (int*)_seqRows.P;
            long slots = _state.Dims[0] / _slotCtx;
            for (long i = 0; i < t; i++)
            {
                if (_pages is not null)
                {
                    _pages.Check(rw[i * 4], rw[i * 4 + 1], _state.Dims[0]);
                }
                else if ((uint)rw[i * 4] >= slots || (uint)rw[i * 4 + 1] >= _slotCtx)
                {
                    throw new InvalidOperationException($"KV cache overflow: sequence row {i} slot {rw[i * 4]} position {rw[i * 4 + 1]} (slot context {_slotCtx}).");
                }
            }
        }
        else if (!p.Scalars.TryGetValue(_offset, out _past))
        {
            throw new InvalidOperationException($"Runtime scalar '{_offset}' was not provided.");
        }
        else if (_past < 0 || _past + t > _state.Dims[0])
        {
            throw new InvalidOperationException($"KV cache overflow: n_past {_past} + T {t} > context {_state.Dims[0]}.");
        }

        _rows = _x.Rows(p);
        int chunks = (int)Math.Min(_rows, CpuParallel.Chunks(_rows * _rowLen, 8192));
        _per = (_rows + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    private void Body(int c)
    {
        long s = c * _per;
        long e = Math.Min(_rows, s + _per);
        var type = _state.Type;
        for (long r = s; r < e; r++)
        {
            float* src = _x.F(r);
            long dstElem;
            if (_seqRows is not null)
            {
                long tok = r / _rowsPerToken;
                int* rw = (int*)_seqRows.P + tok * 4;
                long row = _pages?.Row(rw[0], rw[1]) ?? rw[0] * _slotCtx + rw[1];
                dstElem = row * _inner + (r - tok * _rowsPerToken) * _rowLen;
            }
            else
            {
                dstElem = _past * _inner + r * _rowLen;
            }

            if (type == DataType.F32)
            {
                Buffer.MemoryCopy(src, (float*)_state.P + dstElem, _rowLen * 4, _rowLen * 4);
            }
            else if (_quantized is { } format)
            {
                Quant.Quantize(format, src, _rowLen, _state.P + dstElem / 32 * Quant.BlockBytes(format));
            }
            else if (type == DataType.F16) { ushort* d = (ushort*)_state.P + dstElem; for (long i = 0; i < _rowLen; i++)
                {
                    d[i] = Simd.FloatToHalf(src[i]);
                }
            }
            else if (type == DataType.BF16) { ushort* d = (ushort*)_state.P + dstElem; for (long i = 0; i < _rowLen; i++)
                {
                    d[i] = Simd.FloatToBFloat(src[i]);
                }
            }
            else
            {
                throw new NotSupportedException($"CacheWrite into {type}");
            }
        }
    }
}
