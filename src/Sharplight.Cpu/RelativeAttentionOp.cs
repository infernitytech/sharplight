using System.Numerics;
using System.Runtime.CompilerServices;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Self-attention with Transformer-XL relative positions over dense f32 rows (see <c>StandardOps.RelativeAttention</c>): content queries
/// qc and position queries qp <c>[T, H, D]</c>, keys <c>[T, H, D]</c>, values <c>[T, H, Dv]</c>, position keys <c>[2T − 1, H, D]</c>.
/// Parallel over (query row, head) pairs; each computes its T scores, the softmax and the weighted sum of the values.
/// </summary>
internal sealed unsafe class RelativeAttentionOp : CpuOp
{
    private readonly TRef _qc;
    private readonly TRef _qp;
    private readonly TRef _k;
    private readonly TRef _v;
    private readonly TRef _p;
    private readonly TRef _y;
    private readonly float _scale;
    private readonly Action<int> _body;
    private long _rows;
    private long _per;

    /// <summary>Creates the op.</summary>
    public RelativeAttentionOp(TRef qc, TRef qp, TRef k, TRef v, TRef p, TRef y, float scale)
    {
        _qc = qc;
        _qp = qp;
        _k = k;
        _v = v;
        _p = p;
        _y = y;
        _scale = scale;
        _body = Body;
    }

    public override string Name => "RelativeAttention";

    public override void Run(RunParameters p)
    {
        _rows = _qc.Dim(0, p);
        if (_rows == 0)
        {
            return;
        }

        long heads = _qc.Dims[1];
        long work = _rows * (_qc.Dims[2] + _v.Dims[2]) * 2;
        var (chunks, per) = OpHelpers.Split(_rows * heads, work);
        _per = per;
        CpuParallel.For(chunks, _body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private void Body(int chunk)
    {
        long heads = _qc.Dims[1];
        int dim = (int)_qc.Dims[2];
        int valueDim = (int)_v.Dims[2];
        long rows = _rows;
        long first = chunk * _per;
        long last = Math.Min(rows * heads, first + _per);
        float* qc = (float*)_qc.P;
        float* qp = (float*)_qp.P;
        float* k = (float*)_k.P;
        float* v = (float*)_v.P;
        float* pos = (float*)_p.P;
        float* y = (float*)_y.P;
        long keyStride = heads * dim;
        long valueStride = heads * valueDim;
        var scores = new float[rows];
        var accumulator = new float[valueDim];
        for (long item = first; item < last; item++)
        {
            long i = item / heads;
            long h = item % heads;
            float* content = qc + (i * keyStride) + (h * dim);
            float* position = qp + (i * keyStride) + (h * dim);
            float max = float.NegativeInfinity;
            for (long j = 0; j < rows; j++)
            {
                float* key = k + (j * keyStride) + (h * dim);
                float* relative = pos + (((rows - 1 - i) + j) * keyStride) + (h * dim);
                float score = _scale * (Dot(content, key, dim) + Dot(position, relative, dim));
                scores[j] = score;
                max = MathF.Max(max, score);
            }

            double sum = 0;
            for (long j = 0; j < rows; j++)
            {
                float e = MathF.Exp(scores[j] - max);
                scores[j] = e;
                sum += e;
            }

            Array.Clear(accumulator);
            for (long j = 0; j < rows; j++)
            {
                float weight = scores[j];
                float* value = v + (j * valueStride) + (h * valueDim);
                for (int d = 0; d < valueDim; d++)
                {
                    accumulator[d] += weight * value[d];
                }
            }

            float inverse = (float)(1 / sum);
            float* output = y + (i * valueStride) + (h * valueDim);
            for (int d = 0; d < valueDim; d++)
            {
                output[d] = accumulator[d] * inverse;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Dot(float* a, float* b, int length)
    {
        int width = Vector<float>.Count;
        var sum = Vector<float>.Zero;
        int d = 0;
        for (; d + width <= length; d += width)
        {
            sum += Unsafe.ReadUnaligned<Vector<float>>(a + d) * Unsafe.ReadUnaligned<Vector<float>>(b + d);
        }

        float result = Vector.Sum(sum);
        for (; d < length; d++)
        {
            result += a[d] * b[d];
        }

        return result;
    }
}
