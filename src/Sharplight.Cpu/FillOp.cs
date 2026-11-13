using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary><c>y[i, ...] = start + i · delta</c> over a dense output whose outer extent may be symbolic (ONNX Range / ConstantOfShape).</summary>
internal sealed unsafe class FillOp(TRef y, double start, double delta) : CpuOp
{
    public override string Name => "Fill";

    public override void Run(RunParameters p)
    {
        long rows = y.Rank == 0 ? 1 : y.Dim(0, p);
        long inner = y.Rank == 0 ? 1 : y.MaxElements / Math.Max(1, y.Dims[0]);
        for (long r = 0; r < rows; r++)
        {
            double v = start + r * delta;
            long offset = r * inner;
            if (y.Type == DataType.I32)
            {
                new Span<int>((int*)y.P + offset, (int)inner).Fill((int)v);
            }
            else if (y.Type == DataType.I64)
            {
                new Span<long>((long*)y.P + offset, (int)inner).Fill((long)v);
            }
            else
            {
                new Span<float>((float*)y.P + offset, (int)inner).Fill((float)v);
            }
        }
    }
}
