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
        bool parallelForm = DiagnosticSwitches.Conv1dParallel;
        // Rollback form: the walk replays every run's committed rows and serves the short runs (recording tentative ones); runs
        // long enough for the token-parallel kernels, and longer than any tentative run, are then computed by those.
        int parallelMin = records is null ? Conv1dParallelMinRows : Math.Max(Conv1dParallelMinRows, (int)records.Shape[2] + 1);
        var macros = Macros([("C", C), ("KS", KS), ("XS", x.Strides[0]), ("ACT_SILU", n.Attributes.Get("activation", "none") == "silu"),
            ("HAS_BIAS", cb is not null), ("MULTI_SEQ", runs is not null), ("TOK", tok), .. rollback,
            ("PAR_MIN", records is not null && parallelForm ? parallelMin : 0)]);
        var k = Lib("recurrent.slang", "conv1d", macros);
        KernelArg[] args = [Pointer(x), Pointer(w), Pointer(st), Pointer(y), Pointer(cb ?? y), .. SeqArgs(runs, commits, records), TokenArg(x),
            .. runs is null ? [] : (KernelArg[])[TokenArg(runs)]];
        TensorRef[] reads = [x, w, st, .. cb is null ? [] : (TensorRef[])[cb], .. SequenceTables(runs, commits, records)];
        TensorRef[] recordWrites = records is null ? [] : [records];
        uint Runs(Bindings b) => runs is null ? 1u : (uint)Rows(b, runs);
        // conv1d_tokens: one block per TOK rows of the batch (multi-sequence blocks look up the run of each row).
        Grid TokenGrid(Bindings b) => new(Grid.Div(C, 128), Grid.Div(Rows(b, x), tok), 1);
        if (!parallelForm)
        {
            Emit($"conv1d {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(C, 128), 1, Runs(b)))], args, reads, [y, st, .. recordWrites]);
            return;
        }

        // Prefill: token-parallel outputs, then the state update (a run walks its tokens in turn only for short batches,
        // where the extra launch would cost more than the walk).
        Func<Bindings, bool> parallel = runs is null ? b => Rows(b, x) >= parallelMin : b => Rows(b, x) - Rows(b, runs) + 1 >= parallelMin;
        var tokens = Lib("recurrent.slang", "conv1d_tokens", macros);
        var state = Lib("recurrent.slang", "conv1d_state", macros);
        if (records is not null)
        {
            // The walk first (replay, short runs, records), then the long runs from the replayed state.
            Emit($"conv1d {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(C, 128), 1, Runs(b)))], args, reads, [y, st, .. recordWrites]);
            Emit($"conv1d {n.Output.Name}", [n], [new KernelVariant(parallel, tokens, TokenGrid)], args, reads, [y]);
        }
        else
        {
            Emit($"conv1d {n.Output.Name}", [n],
                [new KernelVariant(parallel, tokens, TokenGrid),
                 new KernelVariant(_ => true, k, b => new Grid(Grid.Div(C, 128), 1, Runs(b)))],
                args, reads, [y, st]);
        }

        Emit($"conv1d {n.Output.Name}", [n], [new KernelVariant(parallel, state, b => new Grid(Grid.Div(C, 128), 1, Runs(b)))],
            args, reads, [st]);
    }

    /// <summary>The <c>STATE_FMT</c> of a recurrent state type (<c>state16.slang</c>): 0 f32, 1 f16, 2 bf16.</summary>
    private static int RecurrentStateFormat(DataType type)
        => type == DataType.F32 ? 0 : type == DataType.F16 ? 1 : type == DataType.BF16 ? 2
            : throw new NotSupportedException($"recurrent state type {type}");

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
        // Per-channel decay ([T, HV, DK]), time-invariant gates ([HV]) and the additive rule run token by token only.
        // (From the node's values: dense copies may be flattened.)
        int gRank = n.Inputs[3].Shape.Rank;
        int betaRank = n.Inputs[4].Shape.Rank;
        bool channelDecay = gRank == 3;
        bool additive = n.Attributes.Get("rule", "delta") switch
        {
            "delta" => false,
            "additive" => true,
            var other => throw new NotSupportedException($"GatedDeltaRule: unknown rule '{other}'."),
        };
        bool plainGates = !channelDecay && !additive && gRank == 2 && betaRank == 2;
        // Rollback form: the token-by-token kernel runs first and replays every run's committed rows (and serves the short runs,
        // recording tentative ones); runs long enough for the chunked form, and longer than any tentative run, then take it.
        var chunked = plainGates ? ChunkedDeltaPlan(DK, DV) : null;
        int chunkMin = Math.Max(1, DiagnosticSwitches.DeltaChunkMinRows);
        if (records is not null)
        {
            chunkMin = Math.Max(chunkMin, (int)records.Shape[2] + 1);
        }

        var common = Macros(("HK", HK), ("HV", HV), ("DK", DK), ("DV", DV),
            ("QS0", q.Strides[0]), ("QS1", q.Strides[1]), ("KS0", k.Strides[0]), ("KS1", k.Strides[1]), ("VS0", v.Strides[0]), ("VS1", v.Strides[1]),
            ("SCALE", n.Attributes.Get<float>("scale")),
            ("NORM_Q", normQ is not null), ("EPS_Q", normQ?.Attributes.Get<float>("eps") ?? 0f),
            ("NORM_K", normK is not null), ("EPS_K", normK?.Attributes.Get<float>("eps") ?? 0f),
            ("MULTI_SEQ", runs is not null), ("CHANNEL_DECAY", channelDecay), ("ADDITIVE", additive),
            ("G_TS", gRank == 1 ? 0 : HV), ("B_TS", betaRank == 1 ? 0 : HV), ("STATE_FMT", RecurrentStateFormat(st.Type)));
        if (chunked is { } plan && records is null)
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
        foreach (var (key, value) in Macros([("LPR", lanesPerRow), ("DVB", dvb), ("TB", Math.Clamp((channelDecay ? 1536 : 2048) / DK, 1, 16)),
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
        if (chunked is { } rollbackPlan && records is not null)
        {
            EmitChunkedDeltaRule(nodes, rollbackPlan, common, chunkMin, q, k, v, g, beta, st, o, runs);
        }
    }
}
