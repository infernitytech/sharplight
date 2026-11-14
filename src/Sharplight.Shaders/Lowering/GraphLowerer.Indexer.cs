using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Lightning-indexer key selection (IndexerTopK, kernels in indexer.slang).</summary>
public sealed partial class GraphLowerer
{
    /// <summary>Threads per workgroup of the indexer kernels (one key position per thread when scoring; the radix selection needs 256).</summary>
    private const int IndexerWorkgroup = 256;

    /// <summary>Largest score scratch of one launch; longer batches are split into launches of fewer rows.</summary>
    private const long IndexerScratchBytes = 64L << 20;

    private BufferDecl? _indexerScratch;

    /// <summary>
    /// Scores every visible key of each query row into a scratch row of the per-sequence context, then selects the best k per row. Batches whose
    /// score rows exceed <see cref="IndexerScratchBytes"/> run in several launches of fewer rows (skipped when the batch is shorter).
    /// </summary>
    private void LowerIndexerTopK(Node n)
    {
        var q = EnsureDense(AsF32(Ref(n.Inputs[0]), "indexer-q"), "indexer-q");
        var keys = Ref(n.Inputs[1]);
        var weights = EnsureDense(AsF32(Ref(n.Inputs[2]), "indexer-w"), "indexer-w");
        var output = Produce(n.Output);
        var seq = n.Inputs.Length > 3 ? DenseInput(n.Inputs[3], "seq-rows") : null;
        var pages = PageTable(n, 5);
        long heads = q.Shape[1];
        long dims = q.Shape[2];
        long k = n.Attributes.Get<long>("k");
        long context = pages is not null ? pages.Shape[1] * n.Attributes.Get("block_size", 0L) : seq is not null ? SlotContext(n, keys) : keys.Shape[0];
        long maxRows = q.Symbol is null ? q.Shape[0] : Bound(Dim.Sym(q.Symbol));
        long sliceRows = Math.Clamp(IndexerScratchBytes / (context * 4), 1, maxRows);

        _indexerScratch ??= NewBuffer(BufferKind.Activations, "indexer-scores", 0);
        _indexerScratch.Bytes = Math.Max(_indexerScratch.Bytes, sliceRows * context * 4);
        var scores = new TensorRef(_indexerScratch, 0, DataType.F32, [sliceRows * context], [1], null);

        var macros = Macros([.. PageMacros(n, pages), ("MULTI_SEQ", seq is not null), ("SLOT_CTX", seq is null ? 0 : SlotContext(n, keys)),
            ("H", heads), ("D", dims), ("K", k), ("CTX", context), ("KEY_F16", keys.Type == DataType.F16), ("WG", IndexerWorkgroup),
            ("RT_N_PAST", seq is null ? Slot(n) : 0)]);
        var scoreKernel = Lib("indexer.slang", "index_scores", macros);
        var selectKernel = Lib("indexer.slang", "index_select", macros);
        for (long firstRow = 0; firstRow < maxRows; firstRow += sliceRows)
        {
            long first = firstRow;
            KernelArg[] args = [Pointer(q), Pointer(keys), Pointer(weights), Pointer(scores), Pointer(output), new PointerArg(_runtime, 0),
                .. SeqArgs(seq, pages), TokenArg(q), new UIntArg((uint)first)];
            TensorRef[] tables = PresentTables(seq, pages);
            Func<Bindings, long> rows = b => Math.Min(sliceRows, Rows(b, q) - first);
            Emit($"indexer scores {n.Output.Name}", [n],
                [new KernelVariant(b => Rows(b, q) > first, scoreKernel, b => new Grid(Grid.Div(context, IndexerWorkgroup), (uint)rows(b)))],
                args, [q, keys, weights, .. tables], [scores]);
            Emit($"indexer select {n.Output.Name}", [n],
                [new KernelVariant(b => Rows(b, q) > first, selectKernel, b => new Grid((uint)rows(b)))],
                args, [scores, .. tables], [output]);
        }
    }
}
