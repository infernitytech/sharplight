using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Rows per chunk of the chunked delta rule (fixed by <c>chunked.slang</c>).</summary>
    private const int DeltaChunkRows = 64;

    /// <summary>Scratch shared by every chunked delta-rule launch (the layers run one after another).</summary>
    private BufferDecl? _deltaChunkScratch;

    /// <summary>
    /// Operand parts (1 = f16, 2 = f16 hi + lo), state columns per workgroup and threads per workgroup of the sequential pass
    /// of the chunked delta rule.
    /// </summary>
    private readonly record struct DeltaChunkPlan(int Parts, int Columns, int StateThreads);

    /// <summary>
    /// The chunked (WY-form, tensor-core) gated delta rule for long runs, or null when the device or the head sizes do not
    /// allow it. Needs f16 cooperative matrices on 32-wide subgroups, head sizes that are multiples of 16 and the tiles in
    /// shared memory. Precision: f32 activations get the two-part f16 split (about 22 mantissa bits) unless reduced
    /// precision is allowed, which takes one f16 product per step.
    /// </summary>
    private DeltaChunkPlan? ChunkedDeltaPlan(long dk, long dv)
    {
        if (!DiagnosticSwitches.DeltaChunked || !_opt.Has(DeviceFeatures.MatrixF16) || !_opt.Has(DeviceFeatures.Float16)
            || !_opt.Has(DeviceFeatures.SubgroupOps) || _opt.SubgroupSize != 32 || dk % 16 != 0 || dv % 16 != 0 || dk > 256)
        {
            return null;
        }

        int parts = DiagnosticSwitches.DeltaChunkParts is 1 or 2 ? DiagnosticSwitches.DeltaChunkParts : _opt.Compile.AllowReducedPrecision ? 1 : 2;
        int[] candidates = DiagnosticSwitches.DeltaChunkColumns > 0 ? [DiagnosticSwitches.DeltaChunkColumns] : [32, 16];
        foreach (int columns in candidates)
        {
            if (columns % 16 == 0 && dv % columns == 0 && DeltaChunkSharedBytes(dk, columns, parts) <= _opt.MaxSharedMemory
                && DeltaChunkPrepSharedBytes() <= _opt.MaxSharedMemory)
            {
                // Four subgroups for the two-part form on 32 columns: fewer registers, so two workgroups fit on a
                // multiprocessor, which outweighs the halved parallelism inside one (27B at T = 2048: 520 vs 609 us); the
                // one-part form and 16 columns are faster with eight (242 vs 290 us).
                int threads = DiagnosticSwitches.DeltaChunkThreads is 128 or 256 ? DiagnosticSwitches.DeltaChunkThreads : parts == 2 && columns >= 32 ? 128 : 256;
                return new DeltaChunkPlan(parts, columns, threads);
            }
        }

        return null;
    }

    /// <summary>Shared memory of <c>delta_chunk_state</c>: f16 state / U' planes, R / U planes, f32 staging and G.</summary>
    private static long DeltaChunkSharedBytes(long dk, int columns, int parts)
    {
        long rowStride = columns + 8;
        return parts * Math.Max(dk, DeltaChunkRows) * rowStride * 2 + parts * DeltaChunkRows * rowStride * 2
               + Math.Max(2L * DeltaChunkRows, dk) * columns * 4 + DeltaChunkRows * 4;
    }

    /// <summary>Shared memory of <c>delta_chunk_prep</c>: two f32 64x68 tiles, G and beta.</summary>
    private static long DeltaChunkPrepSharedBytes() => 2L * DeltaChunkRows * (DeltaChunkRows + 4) * 4 + 2 * DeltaChunkRows * 4;

    /// <summary>
    /// Emits the three launches of the chunked delta rule (<c>chunked.slang</c>): normalized f16 q / k planes, the per-chunk
    /// triangular solve (all chunks in parallel), and the pass over the chunks that carries the state. They run when a
    /// sequence can have at least <paramref name="chunkMin"/> rows; multi-sequence batches serve the shorter runs with the
    /// token-by-token kernel in the same step.
    /// </summary>
    private void EmitChunkedDeltaRule(Node[] nodes, DeltaChunkPlan plan, Dictionary<string, string> common, int chunkMin,
        TensorRef q, TensorRef k, TensorRef v, TensorRef g, TensorRef beta, TensorRef st, TensorRef o, TensorRef? runs)
    {
        long hk = q.Shape[1];
        long dk = q.Shape[2];
        long hv = v.Shape[1];
        long dv = v.Shape[2];
        long scratchRows = q.Shape[0] + DeltaChunkRows;
        long qkBytes = Align256(plan.Parts * 2L * scratchRows * hk * dk * 2);
        long tpBytes = Align256(plan.Parts * 2L * scratchRows * hv * DeltaChunkRows * 2);
        long gBytes = Align256(scratchRows * hv * 4);
        _deltaChunkScratch ??= NewBuffer(BufferKind.Activations, "delta-chunk", 0);
        _deltaChunkScratch.Bytes = Math.Max(_deltaChunkScratch.Bytes, qkBytes + tpBytes + gBytes);
        var qkPlanes = new TensorRef(_deltaChunkScratch, 0, DataType.F16, [qkBytes / 2], [1], null);
        var tpPlanes = new TensorRef(_deltaChunkScratch, qkBytes, DataType.F16, [tpBytes / 2], [1], null);
        var cumDecay = new TensorRef(_deltaChunkScratch, qkBytes + tpBytes, DataType.F32, [gBytes / 4], [1], null);

        var macros = new Dictionary<string, string>(common, StringComparer.Ordinal);
        foreach (var (key, value) in Macros(("PARTS", plan.Parts), ("BV", plan.Columns), ("STATE_THREADS", plan.StateThreads), ("SROWS", scratchRows), ("CHUNK_MIN", runs is null ? 0 : chunkMin)))
        {
            macros[key] = value;
        }

        // A run of chunkMin rows exists only if the batch has that many rows beyond one per other run.
        Func<Bindings, bool> when = runs is null ? b => Rows(b, q) >= chunkMin : b => Rows(b, q) - Rows(b, runs) + 1 >= chunkMin;
        uint Runs(Bindings b) => runs is null ? 1u : (uint)Rows(b, runs);
        KernelArg[] args = [Pointer(q), Pointer(k), Pointer(v), Pointer(g), Pointer(beta), Pointer(st), Pointer(o),
            Pointer(qkPlanes), Pointer(tpPlanes), Pointer(cumDecay), .. SeqArgs(runs), TokenArg(q)];
        TensorRef[] runTable = runs is null ? [] : [runs];
        string name = nodes[^1].Output.Name;

        var qkKernel = Lib("chunked.slang", "delta_chunk_qk", macros);
        Emit($"delta_chunk {name}", nodes, [new KernelVariant(when, qkKernel, b => new Grid(Grid.Div(Rows(b, q) + DeltaChunkRows, 4), (uint)hk))],
            args, [q, k], [qkPlanes]);

        var prepKernel = Lib("chunked.slang", "delta_chunk_prep", macros);
        Emit($"delta_chunk {name}", nodes, [new KernelVariant(when, prepKernel, b => new Grid((uint)hv, Grid.Div(Rows(b, q), DeltaChunkRows), Runs(b)))],
            args, [qkPlanes, g, beta, .. runTable], [tpPlanes, cumDecay]);

        var stateKernel = Lib("chunked.slang", "delta_chunk_state", macros);
        Emit($"delta_chunk {name}", nodes, [new KernelVariant(when, stateKernel, b => new Grid((uint)hv, (uint)(dv / plan.Columns), Runs(b)))],
            args, [qkPlanes, tpPlanes, cumDecay, v, st, .. runTable], [o, st]);
    }

    private static long Align256(long bytes) => (bytes + 255) / 256 * 256;
}
