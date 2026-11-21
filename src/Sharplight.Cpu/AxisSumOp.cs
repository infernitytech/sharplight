using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// ReduceSum over the last axis of a [A, H, K] view whose H axis is contiguous and whose K axis is not (a permuted [A, K, H] tensor, such as
/// the slot sum of routed experts, <c>ReduceSum(Permute(x, 0, 2, 1))</c>): the K rows of H values are added as vectors, without first
/// copying the view into rows of K.
/// </summary>
internal sealed unsafe class AxisSumOp : CpuOp
{
    /// <summary>Values of H per work item.</summary>
    private const long Block = 1024;

    private readonly TRef _x;
    private readonly TRef _y;
    private readonly long _h;
    private readonly long _k;
    private readonly Action<int> _body;
    private long _items;
    private readonly long _blocks;
    private long _per;

    public AxisSumOp(TRef x, TRef y)
    {
        _x = x;
        _y = y;
        _h = x.Dims[1];
        _k = x.Dims[2];
        _blocks = (_h + Block - 1) / Block;
        _body = Body;
    }

    public override string Name => "AxisSum";

    public override void Run(RunParameters p)
    {
        _items = _x.Dim(0, p) * _blocks;
        if (_items == 0)
        {
            return;
        }

        int chunks = CpuParallel.Chunks(_items * _h / _blocks * _k, 32768);
        chunks = (int)Math.Min(chunks, _items);
        _per = (_items + chunks - 1) / chunks;
        CpuParallel.For(chunks, _body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int c)
    {
        long end = Math.Min(_items, (c + 1) * _per);
        long strideA = _x.Strides[0];
        long strideK = _x.Strides[2];
        for (long item = c * _per; item < end; item++)
        {
            long a = item / _blocks;
            long h0 = item % _blocks * Block;
            long n = Math.Min(Block, _h - h0);
            float* source = (float*)_x.P + a * strideA + h0;
            float* destination = (float*)_y.P + a * _h + h0;
            long i = 0;
            for (; Vector256.IsHardwareAccelerated && i + 8 <= n; i += 8)
            {
                var sum = Vector256.Load(source + i);
                for (long j = 1; j < _k; j++)
                {
                    sum += Vector256.Load(source + j * strideK + i);
                }

                sum.Store(destination + i);
            }

            for (; Vector128.IsHardwareAccelerated && i + 4 <= n; i += 4)
            {
                var sum = Vector128.Load(source + i);
                for (long j = 1; j < _k; j++)
                {
                    sum += Vector128.Load(source + j * strideK + i);
                }

                sum.Store(destination + i);
            }

            for (; i < n; i++)
            {
                float sum = source[i];
                for (long j = 1; j < _k; j++)
                {
                    sum += source[j * strideK + i];
                }

                destination[i] = sum;
            }
        }
    }
}
