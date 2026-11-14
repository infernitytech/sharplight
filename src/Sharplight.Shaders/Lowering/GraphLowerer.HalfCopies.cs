using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// f16 copies of root values written by the row norm that produced them, with the step row counts for which the norm
    /// writes the copy; <see cref="QuantizedActivations"/> hands them to the tensor-core GEMMs instead of launching a
    /// conversion (and converts into the same buffer for larger steps).
    /// </summary>
    private readonly Dictionary<Value, (TensorRef Half, long MinRows, long MaxRows)> _halfCopies = [];

    /// <summary>Root values read as x by a matmul over non-f32 weights (built on first use).</summary>
    private HashSet<Value>? _halfMatMulRoots;

    /// <summary>
    /// Largest step for which a row norm writes the f16 copy: the skinny GEMM range, where the conversion launches are a
    /// visible share of the step (Blackwell, 27B FP8 at 16 rows: -0.8 ms per step). Larger steps convert separately, since
    /// the extra norm stores cost more than the launch saves there (171-row mixed step: +1.3 ms when the norm writes it).
    /// </summary>
    private static long NormHalfCopyMaxRows => SkinnyMaxRows;

    /// <summary>
    /// Adds an f16 copy output to a row norm for each pass-2 output that a tensor-core matmul reads as x: the norm stores the
    /// same value rounded to f16 next to the f32 row (bit-identical to the separate conversion) for steps of
    /// <see cref="NormHalfCopyMaxRows"/> rows or fewer where some GEMM reads an f16 copy. Needs f16 tensor-core matrices;
    /// <c>SHARPLIGHT_NORM_HALF_COPY=0</c> disables it.
    /// </summary>
    private void AddHalfCopyOutputs(IReadOnlyList<Value>? values, List<(TensorRef Ref, string Expr)> pass2Outputs,
        Dictionary<int, (long Min, long Max)> rowRanges)
    {
        if (values is null || !DiagnosticSwitches.NormHalfCopy || !_opt.Has(DeviceFeatures.MatrixF16) || !_opt.Has(DeviceFeatures.Float16))
        {
            return;
        }

        long minRows = Math.Min(SkinnyHalfMinRows, MatvecMaxRows + 1);
        long maxRows = NormHalfCopyMaxRows;
        if (minRows > maxRows)
        {
            return;
        }

        int count = pass2Outputs.Count;
        for (int i = 0; i < count && i < values.Count; i++)
        {
            var value = values[i];
            var y = pass2Outputs[i].Ref;
            long k = y.Shape.Length == 0 ? 0 : y.Shape[^1];
            if (value.Type != DataType.F32 || y.Type != DataType.F32 || !y.IsDense || y.Buffer != _arena || k % 8 != 0
                || RootOf(value) != value || _halfCopies.ContainsKey(value) || !HalfMatMulRoots().Contains(value))
            {
                continue;
            }

            var allocation = new object();
            long elements = y.Shape.Aggregate(1L, (a, d) => a * d);
            long offset = _alloc.Allocate(allocation, DataType.F16.ByteSize(elements));
            var half = new TensorRef(_arena, offset, DataType.F16, y.Shape, TensorRef.DenseStrides(y.Shape), y.Symbol);
            if (!_rootScratch.TryGetValue(value, out var list))
            {
                _rootScratch[value] = list = [];
            }

            list.Add(allocation);
            _halfCopies[value] = (half, minRows, maxRows);
            rowRanges[pass2Outputs.Count] = (minRows, maxRows);
            pass2Outputs.Add((half, pass2Outputs[i].Expr));
        }
    }

    /// <summary>
    /// The norm-written f16 copy of <paramref name="root"/> (as f16 rows of <paramref name="k"/> values) and the largest step
    /// the norm writes it for, when it serves a conversion of x (the root's own dense rows) for steps from
    /// <paramref name="minRows"/> rows; else null.
    /// </summary>
    private (TensorRef Copy, long MaxRows)? NormHalfCopy(Value root, TensorRef x, long xStride, long k, long minRows)
    {
        if (!_halfCopies.TryGetValue(root, out var copy) || minRows < copy.MinRows || xStride != k || copy.Half.Shape[^1] != k
            || x.ByteOffset != Ref(root).ByteOffset)
        {
            return null;
        }

        long rows = copy.Half.Shape.Aggregate(1L, (a, d) => a * d) / k;
        long[] shape = [rows, 2 * k];
        return (new TensorRef(_arena, copy.Half.ByteOffset, DataType.U8, shape, TensorRef.DenseStrides(shape), copy.Half.Symbol), copy.MaxRows);
    }

    private HashSet<Value> HalfMatMulRoots()
    {
        if (_halfMatMulRoots is null)
        {
            _halfMatMulRoots = [];
            foreach (var node in _seg.Nodes)
            {
                if (node.Op.Name == "MatMul" && node.Inputs.Length > 1 && node.Inputs[1].Type != DataType.F32)
                {
                    _halfMatMulRoots.Add(RootOf(node.Inputs[0]));
                }
            }
        }

        return _halfMatMulRoots;
    }
}
