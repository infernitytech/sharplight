using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>
    /// f16 copies of root values written by the launch that produced them (a row norm, the dual gemm_cm2), with the step row
    /// counts for which it writes the copy (from MinRows, up to MaxRows where Written is null, else where Written holds);
    /// <see cref="QuantizedActivations"/> hands them to the tensor-core GEMMs instead of launching a conversion (and converts
    /// into the same buffer for the other steps).
    /// </summary>
    private readonly Dictionary<Value, (TensorRef Half, long MinRows, long MaxRows, Func<long, bool>? Written)> _halfCopies = [];

    /// <summary>Root values read as x by a matmul over non-f32 weights (built on first use).</summary>
    private HashSet<Value>? _halfMatMulRoots;

    /// <summary>
    /// Largest step for which a row norm writes the f16 copy next to the f32 row: the skinny GEMM range, where the conversion
    /// launches are a visible share of the step (Blackwell, 27B FP8 at 16 rows: -0.8 ms per step). Larger steps convert
    /// separately, since the extra norm stores cost more than the launch saves there (171-row mixed step: +1.3 ms when the norm
    /// writes it), up to <see cref="NormCopyPrefillMinRows"/>, from where the norm writes only the copy.
    /// </summary>
    private static long NormHalfCopyMaxRows => SkinnyMaxRows;

    /// <summary>
    /// Smallest step from which a row norm whose every reader is a tensor-core GEMM writes that GEMM's activations (the f16 copy
    /// or the E4M3 rows of the fp8 GEMM) instead of the f32 row (<c>SHARPLIGHT_NORM_COPY_MIN_ROWS</c>).
    /// </summary>
    private static long NormCopyPrefillMinRows => DiagnosticSwitches.NormCopyMinRows ?? 128;

    /// <summary>E4M3 activation rows written by a row norm for steps of MinRows rows or more (see <see cref="AddFp8CopyOutput"/>).</summary>
    private readonly Dictionary<Value, (TensorRef Copy, long MinRows)> _fp8Copies = [];

    /// <summary>
    /// GEMM launches that read a norm-written copy of a root value (by root): the nodes and the step row counts the copy serves
    /// them, with the largest row count of their x (see <see cref="ReadersUseCopy"/>).
    /// </summary>
    private readonly Dictionary<Value, List<(Node[] Nodes, long MinRows, long MaxRows, long RowsMax)>> _copyReaders = [];

    /// <summary>Decisions taken once every launch is lowered (whether row norms may skip their f32 rows).</summary>
    private readonly List<Action> _deferredCopyChecks = [];

    /// <summary>
    /// The activation form the GEMMs reading <paramref name="value"/> as x take at prefill sizes when they all agree on one
    /// that a row norm can write: <see cref="ActivationCode.F16"/> (tensor-core GEMMs over 16-bit or decoded quantized weights),
    /// <see cref="ActivationCode.Fp8"/> (the fp8 GEMM); else null. A prediction: whether a reader actually takes the copy is
    /// checked after lowering (<see cref="ReadersUseCopy"/>) before the norm skips its f32 rows.
    /// </summary>
    private ActivationCode? PrefillCopyCode(Value value, long k)
    {
        if (!DiagnosticSwitches.NormPrefillCopy || !_opt.Has(DeviceFeatures.MatrixF16) || !_opt.Has(DeviceFeatures.Float16))
        {
            return null;
        }

        ActivationCode? code = null;
        bool any = false;
        foreach (var node in _seg.Nodes)
        {
            for (int i = 0; i < node.Inputs.Length; i++)
            {
                if (RootOf(node.Inputs[i]) != value)
                {
                    continue;
                }

                if (node.Op.Name != "MatMul" || i != 0 || node.Inputs.Length < 2 || node.Inputs[1].Type == DataType.F32)
                {
                    return null;
                }

                // The weights' stored type (repacked formats such as Q8_0R decide the GEMM), else the graph's.
                var w = node.Inputs[1];
                var type = _refs.TryGetValue(w, out var stored) ? stored.Type : w.Type;
                long n = w.Shape.Rank > 0 ? w.Shape.Dims[0].Fixed : 0;
                ActivationCode reader = UseFp8Gemm(type, k) && !UseMmq(type, n, k) ? ActivationCode.Fp8
                    : UseMmq(type, n, k) ? ActivationCode.Q8_1 : ActivationCode.F16;
                if (reader == ActivationCode.Q8_1 || (code is { } c && c != reader))
                {
                    return null;
                }

                code = reader;
                any = true;
            }
        }

        return any ? code : null;
    }

    /// <summary>
    /// Adds an f16 copy output to a row norm for each pass-2 output that a tensor-core matmul reads as x: the norm stores the
    /// same value rounded to f16 next to the f32 row (bit-identical to the separate conversion) for steps of
    /// <see cref="NormHalfCopyMaxRows"/> rows or fewer where some GEMM reads an f16 copy, and from
    /// <see cref="NormCopyPrefillMinRows"/> rows where every reader is an f16 tensor-core GEMM (<paramref name="prefill"/> gets
    /// those values). Needs f16 tensor-core matrices; <c>SHARPLIGHT_NORM_HALF_COPY=0</c> disables it.
    /// </summary>
    private void AddHalfCopyOutputs(IReadOnlyList<Value>? values, List<(TensorRef Ref, string Expr)> pass2Outputs,
        Dictionary<int, string> guards, List<Value> prefill)
    {
        if (values is null || !DiagnosticSwitches.NormHalfCopy || !_opt.Has(DeviceFeatures.MatrixF16) || !_opt.Has(DeviceFeatures.Float16))
        {
            return;
        }

        long minRows = Math.Min(SkinnyHalfMinRows, MatvecMaxRows + 1);
        long maxRows = NormHalfCopyMaxRows;
        long prefillRows = NormCopyPrefillMinRows;
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

            bool small = minRows <= maxRows;
            bool large = PrefillCopyCode(value, k) == ActivationCode.F16;
            if (!small && !large)
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
            bool Written(long rows) => (small && rows >= minRows && rows <= maxRows) || (large && rows >= prefillRows);
            _halfCopies[value] = (half, minRows, maxRows, large ? Written : null);
            string smallGuard = $"(p.rows >= {minRows}u && p.rows <= {maxRows}u)";
            guards[pass2Outputs.Count] = !large ? smallGuard : small ? $"{smallGuard} || p.rows >= {prefillRows}u" : $"p.rows >= {prefillRows}u";
            pass2Outputs.Add((half, pass2Outputs[i].Expr));
            if (large)
            {
                prefill.Add(value);
            }
        }
    }

    /// <summary>
    /// Adds an output of E4M3 activation rows (the fp8 GEMM's x, see <see cref="RowNormKernelSource.Fp8Output"/>) for the first
    /// pass-2 value every reader of which is an fp8 GEMM, written for steps of <see cref="NormCopyPrefillMinRows"/> rows or more
    /// (smaller steps quantize separately into the same rows). Returns the output's index, or null.
    /// </summary>
    private int? AddFp8CopyOutput(IReadOnlyList<Value>? values, List<(TensorRef Ref, string Expr)> pass2Outputs, int count, List<Value> prefill)
    {
        if (values is null || !_opt.Has(DeviceFeatures.SubgroupOps) || _opt.SubgroupSize != 32)
        {
            return null;
        }

        for (int i = 0; i < count && i < values.Count; i++)
        {
            var value = values[i];
            var y = pass2Outputs[i].Ref;
            long k = y.Shape.Length == 0 ? 0 : y.Shape[^1];
            // The row passes through shared memory (4 bytes per value, next to the norm's own 1 KB).
            if (value.Type != DataType.F32 || y.Type != DataType.F32 || !y.IsDense || y.Buffer != _arena || k % 128 != 0
                || k * 4 + 1024 > _opt.MaxSharedMemory || RootOf(value) != value || _fp8Copies.ContainsKey(value)
                || PrefillCopyCode(value, k) != ActivationCode.Fp8)
            {
                continue;
            }

            long rows = y.Shape.Aggregate(1L, (a, d) => a * d) / k;
            var allocation = new object();
            long offset = _alloc.Allocate(allocation, rows * (k + k / 32));
            if (!_rootScratch.TryGetValue(value, out var list))
            {
                _rootScratch[value] = list = [];
            }

            list.Add(allocation);
            long[] shape = [rows, k + k / 32];
            _fp8Copies[value] = (new TensorRef(_arena, offset, DataType.U8, shape, TensorRef.DenseStrides(shape), y.Symbol), NormCopyPrefillMinRows);
            long[] words = [rows, (k + k / 32) / 4];
            pass2Outputs.Add((new TensorRef(_arena, offset, DataType.U32, words, TensorRef.DenseStrides(words), y.Symbol), pass2Outputs[i].Expr));
            prefill.Add(value);
            return pass2Outputs.Count - 1;
        }

        return null;
    }

    /// <summary>
    /// The norm-written E4M3 rows of <paramref name="root"/> and the smallest step the norm writes them for, when they serve x
    /// (the root's own dense rows of <paramref name="k"/> values); else null.
    /// </summary>
    private (TensorRef Copy, long MinRows)? NormFp8Copy(Value root, TensorRef x, long xStride, long k)
        => _fp8Copies.TryGetValue(root, out var copy) && xStride == k && copy.Copy.Shape[1] == k + k / 32 && x.ByteOffset == Ref(root).ByteOffset
            ? copy : null;

    /// <summary>Notes that <paramref name="nodes"/> read <paramref name="result"/> as x for steps of <paramref name="minRows"/> to <paramref name="maxRows"/> rows, when it is a norm-written copy of <paramref name="root"/>.</summary>
    private void NoteCopyReader(Value root, TensorRef result, Node[] nodes, long minRows, long maxRows, long rowsMax)
    {
        bool half = _halfCopies.TryGetValue(root, out var h) && h.Half.ByteOffset == result.ByteOffset;
        bool fp8 = _fp8Copies.TryGetValue(root, out var f) && f.Copy.ByteOffset == result.ByteOffset;
        if (result.Buffer != _arena || (!half && !fp8))
        {
            return;
        }

        if (!_copyReaders.TryGetValue(root, out var readers))
        {
            _copyReaders[root] = readers = [];
        }

        readers.Add((nodes, minRows, maxRows, rowsMax));
    }

    /// <summary>
    /// True when every node of the segment reading <paramref name="value"/> (or a view of it) is a GEMM that takes its
    /// norm-written copy for all steps of <paramref name="fromRows"/> rows or more, and the value is not a segment output: the
    /// norm then need not write the f32 rows for those steps.
    /// </summary>
    private bool ReadersUseCopy(Value value, long fromRows)
    {
        if (_seg.Outputs.Any(o => RootOf(o) == value) || !_copyReaders.TryGetValue(value, out var readers))
        {
            return false;
        }

        foreach (var node in _seg.Nodes)
        {
            if (node.Inputs.Any(v => RootOf(v) == value)
                && !readers.Any(r => r.Nodes.Contains(node) && r.MinRows <= fromRows && r.MaxRows >= r.RowsMax))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// The norm-written f16 copy of <paramref name="root"/> (as f16 rows of <paramref name="k"/> values) and the largest step
    /// the norm writes it for, when it serves a conversion of x (the root's own dense rows) for steps from
    /// <paramref name="minRows"/> rows; else null.
    /// </summary>
    private (TensorRef Copy, long MaxRows, Func<long, bool>? Written)? NormHalfCopy(Value root, TensorRef x, long xStride, long k, long minRows)
    {
        if (!_halfCopies.TryGetValue(root, out var copy) || (copy.Written is null && minRows < copy.MinRows) || xStride != k || copy.Half.Shape[^1] != k
            || x.ByteOffset != Ref(root).ByteOffset)
        {
            return null;
        }

        long rows = copy.Half.Shape.Aggregate(1L, (a, d) => a * d) / k;
        long[] shape = [rows, 2 * k];
        return (new TensorRef(_arena, copy.Half.ByteOffset, DataType.U8, shape, TensorRef.DenseStrides(shape), copy.Half.Symbol), copy.MaxRows, copy.Written);
    }

    /// <summary>
    /// An f16 copy of <paramref name="value"/> (the output <paramref name="y"/> of a launch about to be emitted) for that launch to
    /// write for the step row counts where <paramref name="written"/> holds (other steps convert into it); null unless a
    /// tensor-core matmul over non-f32 weights reads the value as x (see <see cref="AddHalfCopyOutputs"/>).
    /// </summary>
    private TensorRef? ProducerHalfCopy(Value value, TensorRef y, Func<long, bool> written)
    {
        long k = y.Shape.Length == 0 ? 0 : y.Shape[^1];
        if (!DiagnosticSwitches.NormHalfCopy || value.Type != DataType.F32 || y.Type != DataType.F32 || !y.IsDense || y.Buffer != _arena
            || k % 8 != 0 || RootOf(value) != value || _halfCopies.ContainsKey(value) || !HalfMatMulRoots().Contains(value))
        {
            return null;
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
        _halfCopies[value] = (half, 0, long.MaxValue, written);
        return half;
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
