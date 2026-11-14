using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Rows of a sequence from which the causal conv1d runs token-parallel (shorter runs walk their tokens).</summary>
    private const int Conv1dParallelMinRows = 16;

    // ---- recurrent ---------------------------------------------------------------------------
    private void LowerConv(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var w = Ref(n.Inputs[1]);
        var st = Ref(n.Inputs[2]);
        if (x.Rank != 2 || x.Strides[1] != 1)
        {
            x = DenseTemp(x, "conv");
        }

        if (w.Type != DataType.F32)
        {
            throw new NotSupportedException("conv weight must be f32");
        }

        var y = Produce(n.Output);
        long C = x.Shape[1];
        long KS = w.Shape[1];
        // Optional bias (attribute "bias", or a non-i32 input 3) and, in the multi-sequence form, the trailing i32 sequence-runs table.
        bool hasBias = n.Attributes.Get("bias", false) || (n.Inputs.Length > 3 && n.Inputs[3].Type != DataType.I32);
        TensorRef? cb = hasBias ? Ref(n.Inputs[3]) : null;
        var runs = RecurrentRuns(n, hasBias ? 4 : 3);
        var (commits, records, rollback) = RollbackBindings(n);
        int tok = Math.Max(1, DiagnosticSwitches.Conv1dTokens);
        var macros = Macros([("C", C), ("KS", KS), ("XS", x.Strides[0]), ("ACT_SILU", n.Attributes.Get("activation", "none") == "silu"),
            ("HAS_BIAS", cb is not null), ("MULTI_SEQ", runs is not null), ("TOK", tok), .. rollback]);
        var k = Lib("recurrent.slang", "conv1d", macros);
        KernelArg[] args = [Pointer(x), Pointer(w), Pointer(st), Pointer(y), Pointer(cb ?? y), .. SeqArgs(runs, commits, records), TokenArg(x)];
        TensorRef[] reads = [x, w, st, .. cb is null ? [] : (TensorRef[])[cb], .. SequenceTables(runs, commits, records)];
        uint Runs(Bindings b) => runs is null ? 1u : (uint)Rows(b, runs);
        // The rollback form (replayed and recorded rows) keeps the per-channel walk.
        if (!DiagnosticSwitches.Conv1dParallel || records is not null)
        {
            Emit($"conv1d {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(C, 128), 1, Runs(b)))], args, reads,
                [y, st, .. records is null ? [] : (TensorRef[])[records]]);
            return;
        }

        // Prefill: token-parallel outputs, then the state update (a run walks its tokens in turn only for short batches,
        // where the extra launch would cost more than the walk).
        Func<Bindings, bool> parallel = runs is null ? b => Rows(b, x) >= Conv1dParallelMinRows : b => Rows(b, x) - Rows(b, runs) + 1 >= Conv1dParallelMinRows;
        var tokens = Lib("recurrent.slang", "conv1d_tokens", macros);
        var state = Lib("recurrent.slang", "conv1d_state", macros);
        Emit($"conv1d {n.Output.Name}", [n],
            [new KernelVariant(parallel, tokens, b => new Grid(Grid.Div(C, 128), Grid.Div(Rows(b, x), tok), Runs(b))),
             new KernelVariant(_ => true, k, b => new Grid(Grid.Div(C, 128), 1, Runs(b)))],
            args, reads, [y, st]);
        Emit($"conv1d {n.Output.Name}", [n], [new KernelVariant(parallel, state, b => new Grid(Grid.Div(C, 128), 1, Runs(b)))],
            args, reads, [st]);
    }

    private void LowerDeltaRule(Node n)
    {
        // q/k may come straight from an absorbed L2Norm: read its input and normalize in the kernel.
        Node? normQ = _absorbedNorms.GetValueOrDefault(n.Inputs[0]);
        Node? normK = _absorbedNorms.GetValueOrDefault(n.Inputs[1]);
        var q = Ref(normQ?.Inputs[0] ?? n.Inputs[0]);
        var k = Ref(normK?.Inputs[0] ?? n.Inputs[1]);
        var v = Ref(n.Inputs[2]);
        var g = Ref(n.Inputs[3]);
        var beta = Ref(n.Inputs[4]);
        var st = Ref(n.Inputs[5]);
        var runs = RecurrentRuns(n, 6);
        var (commits, records, rollback) = RollbackBindings(n);
        q = EnsureUnitInnerStride(q, 2, "delta-q");
        k = EnsureUnitInnerStride(k, 2, "delta-k");
        v = EnsureUnitInnerStride(v, 2, "delta-v");
        g = EnsureDense(g, "delta-g");
        beta = EnsureDense(beta, "delta-beta");

        var o = Produce(n.Output);
        long HK = q.Shape[1];
        long DK = q.Shape[2];
        long HV = v.Shape[1];
        long DV = v.Shape[2];
        Node[] nodes = [.. new[] { normQ, normK }.OfType<Node>(), n];
        // The rollback form (replayed and recorded rows) keeps the token-by-token kernel for every run.
        var chunked = records is null ? ChunkedDeltaPlan(DK, DV) : null;
        int chunkMin = Math.Max(1, DiagnosticSwitches.DeltaChunkMinRows);
        var common = Macros(("HK", HK), ("HV", HV), ("DK", DK), ("DV", DV),
            ("QS0", q.Strides[0]), ("QS1", q.Strides[1]), ("KS0", k.Strides[0]), ("KS1", k.Strides[1]), ("VS0", v.Strides[0]), ("VS1", v.Strides[1]),
            ("SCALE", n.Attributes.Get<float>("scale")),
            ("NORM_Q", normQ is not null), ("EPS_Q", normQ?.Attributes.Get<float>("eps") ?? 0f),
            ("NORM_K", normK is not null), ("EPS_K", normK?.Attributes.Get<float>("eps") ?? 0f),
            ("MULTI_SEQ", runs is not null));
        if (chunked is { } plan)
        {
            EmitChunkedDeltaRule(nodes, plan, common, chunkMin, q, k, v, g, beta, st, o, runs);
        }

        int lanesPerRow = Math.Min(DiagnosticSwitches.DeltaLanesPerRow, _opt.SubgroupSize);
        while ((DV * lanesPerRow > 1024 || DK % lanesPerRow != 0) && lanesPerRow > 1)
        {
            lanesPerRow /= 2;
        }

        // Split each head's DV state rows over several workgroups (rows are independent) for occupancy.
        long wantedRowsPerBlock = DiagnosticSwitches.DeltaRowsPerBlock;
        long dvb = DV % wantedRowsPerBlock == 0 ? wantedRowsPerBlock : DV;
        // With the chunked form, the token-by-token kernel serves the short runs (multi-sequence) or the short batches.
        var macros = new Dictionary<string, string>(common, StringComparer.Ordinal);
        foreach (var (key, value) in Macros([("LPR", lanesPerRow), ("DVB", dvb), ("TB", Math.Clamp(2048 / DK, 1, 16)),
                     ("SEQ_MAX_ROWS", chunked is not null && runs is not null ? chunkMin - 1 : 0), .. rollback]))
        {
            macros[key] = value;
        }

        var kern = Lib("recurrent.slang", "delta_rule", macros);
        Func<Bindings, bool> when = chunked is null || runs is not null ? _ => true : b => Rows(b, q) < chunkMin;
        Emit($"delta_rule {n.Output.Name}", nodes,
            [new KernelVariant(when, kern, b => new Grid((uint)HV, (uint)(DV / dvb), runs is null ? 1u : (uint)Rows(b, runs)))],
            [Pointer(q), Pointer(k), Pointer(v),
             Pointer(g), Pointer(beta), Pointer(st),
             Pointer(o), .. SeqArgs(runs, commits, records), TokenArg(q)],
            [q, k, v, g, beta, st, .. SequenceTables(runs, commits, records)], [o, st, .. records is null ? [] : (TensorRef[])[records]]);
    }
}
