using System.Globalization;
using System.Numerics;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Queries up to which decode attention splits the keys over several workgroups (flash decoding).</summary>
    private const long SplitMaxRows = 4;

    /// <summary>Multi-sequence query rows (batched decode) up to which split-K decoding replaces the per-head kernel.</summary>
    private const long DecodeMaxSequenceRows = 64;

    private BufferDecl? _attnScratch;

    /// <summary>Index of the runtime scalar holding the cache position of a stateful node.</summary>
    private int Slot(Node n) => _scalarSlots[n.Attributes.Get("offset", WellKnown.PastScalar)];

    /// <summary>Multi-sequence ops: cache rows per sequence slot (cache rows / attribute <c>slots</c>).</summary>
    private static long SlotContext(Node n, TensorRef cache)
    {
        long slots = n.Attributes.Get("slots", 1L);
        if (slots < 1 || cache.Shape[0] % slots != 0)
        {
            throw new NotSupportedException($"{n.Op.Name}: cache rows {cache.Shape[0]} not divisible by {slots} slots.");
        }

        return cache.Shape[0] / slots;
    }

    private TensorRef DenseInput(Value v, string name) => EnsureDense(Ref(v), name);

    private static KernelArg[] SeqArgs(params TensorRef?[] tables) => [.. tables.OfType<TensorRef>().Select(t => (KernelArg)Pointer(t))];

    private void LowerCacheWrite(Node n)
    {
        if (_absorbedRope.TryGetValue(n.Inputs[1], out var rope))
        {
            LowerRope(rope, cacheWrite: n);
            return;
        }

        var state = Ref(n.Inputs[0]);
        var x = Ref(n.Inputs[1]);
        x = PrepareRows3(x, "cache", out long r1, out long s0, out long s1);
        long inner = x.Shape[^1];
        bool f16 = state.Type == DataType.F16;
        if (!f16 && state.Type != DataType.F32)
        {
            throw new NotSupportedException($"Cache type {state.Type}");
        }

        var seq = n.Inputs.Length > 2 ? DenseInput(n.Inputs[2], "seq-rows") : null;
        var k = Lib("data.slang", "cache_write", Macros(("R1", r1), ("INNER", inner), ("S0", s0), ("S1", s1), ("DST_F16", f16), ("RT_N_PAST", seq is null ? Slot(n) : 0),
            ("MULTI_SEQ", seq is not null), ("SLOT_CTX", seq is null ? 0 : SlotContext(n, state))));
        var tx = x;
        Emit($"cache_write {state.Buffer.Name}", [n],
            [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * r1 * inner, 128)))],
            [Pointer(x), Pointer(state), new PointerArg(_runtime, 0), .. SeqArgs(seq), TokenArg(x)],
            seq is null ? [x] : [x, seq], [state]);
    }

    private void LowerAttention(Node n)
    {
        var plan = PlanAttention(n);
        var q = plan.Q;
        var o = plan.Output;

        bool flash = CanUseFlashAttention(plan);
        long maxRows = q.Symbol is null ? q.Shape[0] : Bound(Dim.Sym(q.Symbol));
        long sliceRows = SliceRows(plan, flash, maxRows);
        int slices = (int)((maxRows + sliceRows - 1) / sliceRows);

        // Few queries (decode): one workgroup per (head, query) leaves most of the GPU idle, so the keys are split
        // across several workgroups (flash decoding) and merged by a small combine kernel. Preferred: the GQA-grouped
        // split-K kernel with a run-time split count; otherwise SPLITS fixed key ranges per head.
        var decode = DecodeAttention(plan, flash, sliceRows);
        int splits = (int)Math.Clamp((256 + plan.HQ - 1) / plan.HQ, 1, 32);
        bool split = decode is null && !plan.Full && splits > 1 && DiagnosticSwitches.AttentionSplit;
        var part = decode is not null ? DecodeScratch(plan, decode, maxRows) : split ? SplitScratch(plan, splits) : o;
        var splitMacros = new Dictionary<string, string>(plan.Macros) { ["SPLITS"] = splits.ToString(CultureInfo.InvariantCulture) };

        var scalarKernel = Lib("attention.slang", "main", plan.Macros);
        for (int slice = 0; slice < slices; slice++)
        {
            long firstRow = slice * sliceRows;
            var variants = new List<KernelVariant>();
            if (slice == 0 && decode is not null)
            {
                variants.Add(new KernelVariant(decode.When, Lib("attention.slang", "decode", decode.Macros),
                    b => new Grid((uint)decode.Groups, (uint)Rows(b, q), (uint)decode.MaxSplits(Rows(b, q)))));
            }

            if (slice == 0 && split)
            {
                variants.Add(new KernelVariant(b => Rows(b, q) <= SplitMaxRows, Lib("attention.slang", "main", splitMacros),
                    b => new Grid((uint)plan.HQ, (uint)Rows(b, q), (uint)splits)));
            }

            if (slice == 0 && flash)
            {
                variants.Add(FlashVariant(plan));
            }

            if (slices == 1)
            {
                variants.Add(Always(scalarKernel, b => new Grid((uint)plan.HQ, (uint)Rows(b, q))));
            }
            else
            {
                variants.Add(new KernelVariant(b => Rows(b, q) > firstRow, scalarKernel,
                    b => new Grid((uint)plan.HQ, (uint)Math.Min(sliceRows, Rows(b, q) - firstRow))));
            }

            EmitAttentionLaunch(plan, variants, part, firstRow, split || decode is not null);
        }

        if (decode is not null)
        {
            EmitDecodeCombine(plan, part, decode);
        }
        else if (split)
        {
            EmitAttentionCombine(plan, part, splitMacros);
        }
    }

    /// <summary>Resolves the operands of an Attention / CachedAttention node and builds the kernel macros.</summary>
    private AttentionPlan PlanAttention(Node n)
    {
        // "Attention": bidirectional self-attention over K/V given as plain tensors (vision encoders), every query sees all T keys.
        bool full = n.Op.Name == "Attention";
        var q = Ref(n.Inputs[0]);
        var k = Ref(n.Inputs[1]);
        var v = Ref(n.Inputs[2]);
        if (q.Rank != 3 || q.Strides[2] != 1)
        {
            q = DenseTemp(q, "attn-q");
        }

        if (full)
        {
            k = AsDenseF32(k, "attn-k");
            v = AsDenseF32(v, "attn-v");
        }

        var output = Produce(n.Output);
        bool hasSinks = !full && n.Attributes.Get("sinks", false);
        int sinksInputs = hasSinks ? 1 : 0;   // inputs: q, k, v, [sinks], [rows, runs]
        TensorRef? sinks = hasSinks ? Ref(n.Inputs[3]) : null;   // attention sinks [HQ] f32
        long HQ = q.Shape[1];
        long D = q.Shape[2];
        long HKV = k.Shape[1];
        long valueRowStride = v.Shape[2];
        long DV = n.Attributes.Get("v_dims", 0L) is > 0 and var valueDims ? valueDims : valueRowStride;   // v_dims: values are a row prefix
        if (k.Type != v.Type)
        {
            throw new NotSupportedException("K and V caches must have the same type.");
        }

        int window = n.Attributes.Get("window", 0);
        float softcap = n.Attributes.Get("softcap", 0f);
        TensorRef? seq = null;
        TensorRef? runs = null;
        TensorRef? spans = null;
        bool causal = full && n.Attributes.Get("causal", false);
        if (full && n.Inputs.Length > 3)
        {
            spans = DenseInput(n.Inputs[3], "attn-spans");
        }

        if (!full && n.Inputs.Length > 3 + sinksInputs)
        {
            seq = DenseInput(n.Inputs[3 + sinksInputs], "seq-rows");
            runs = n.Inputs.Length > 4 + sinksInputs
                ? DenseInput(n.Inputs[4 + sinksInputs], "seq-runs")
                : throw new NotSupportedException("Multi-sequence attention needs the runs table.");
        }

        var macros = Macros(
            ("MULTI_SEQ", seq is not null), ("SLOT_CTX", seq is null ? 0 : SlotContext(n, k)), ("HQ", HQ), ("HKV", HKV), ("D", D), ("DV", DV), ("VRS", valueRowStride),
            ("QS0", q.Strides[0]), ("QS1", q.Strides[1]), ("SCALE", n.Attributes.Get("scale", 1f)), ("KV_F16", k.Type == DataType.F16),
            ("FULL_ATTN", full), ("RT_N_PAST", full || seq is not null ? 0 : Slot(n)), ("WINDOW", window), ("CHUNK", n.Attributes.Get("chunk", 0)),
            ("HAS_SOFTCAP", softcap > 0), ("SOFTCAP", softcap > 0 ? softcap : 1f), ("HAS_SINKS", sinks is not null),
            ("SPANS", spans is not null), ("CAUSAL", causal));
        return new AttentionPlan
        {
            Node = n,
            Full = full,
            Q = q,
            K = k,
            V = v,
            Output = output,
            Sinks = sinks,
            Seq = seq,
            Runs = runs,
            Spans = spans,
            HQ = HQ,
            HKV = HKV,
            D = D,
            DV = DV,
            Macros = macros,
        };
    }

    /// <summary>Scratch for the partial results of split-key decode attention: (max, sum, accumulator) per split.</summary>
    private TensorRef SplitScratch(AttentionPlan plan, int splits)
    {
        long rowsMax = plan.Q.Shape[0];
        long bytes = splits * Math.Min(rowsMax, SplitMaxRows) * plan.HQ * (plan.DV + 2) * 4;
        _attnScratch ??= NewBuffer(BufferKind.Activations, "attn-split", 0);
        _attnScratch.Bytes = Math.Max(_attnScratch.Bytes, bytes);
        return new TensorRef(_attnScratch, 0, DataType.F32, [bytes / 4], [1], null);
    }

    /// <summary>
    /// Many queries (prefill, vision encoders): tensor-core flash attention (f16 Q/K/V/P) when the device and shapes allow
    /// it. Head sizes are padded in the kernel; the padded tiles must fit shared memory. f32 caches keep the exact kernel
    /// unless reduced precision is allowed.
    /// </summary>
    private bool CanUseFlashAttention(AttentionPlan plan)
    {
        long paddedD = (plan.D + 15) / 16 * 16;
        long paddedDV = (plan.DV + 63) / 64 * 64;
        long paddedKV = Math.Max(paddedD, paddedDV);
        long sharedBytes = 32 * (paddedD + 8) * 2 + 32 * (paddedKV + 8) * 2 + 32 * 32 * 4 + 32 * 40 * 2 + 32 * 64 * 4;
        bool typeAllowed = plan.K.Type == DataType.F16 || (plan.K.Type == DataType.F32 && (plan.Full || _opt.Compile.AllowReducedPrecision));
        return DiagnosticSwitches.AttentionFlash && _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16)
               && _opt.Has(DeviceFeatures.SubgroupOps) && _opt.SubgroupSize == 32 && typeAllowed
               && plan.D % 8 == 0 && plan.DV % 8 == 0 && paddedD <= 256 && paddedDV <= 256
               && plan.HQ % plan.HKV == 0 && sharedBytes <= _opt.MaxSharedMemory;
    }

    /// <summary>
    /// Devices with a submission budget: the scalar kernel's cost grows with queries x keys, so a long batch is cut into row
    /// slices (launches that are skipped when the batch is shorter) of roughly budget milliseconds each (assuming ~3e9 key-dim
    /// multiply-adds per second, a slow integrated GPU; a faster device just gets a few extra cheap launches).
    /// </summary>
    private long SliceRows(AttentionPlan plan, bool flash, long maxRows)
    {
        if (flash || _opt.SubmissionBudgetMs <= 0)
        {
            return maxRows;
        }

        long maxKeys = plan.Full ? maxRows : _opt.Compile.ContextLength;
        long workPerRow = Math.Max(1, plan.HQ * maxKeys * (plan.D + plan.DV));
        return Math.Clamp((long)(_opt.SubmissionBudgetMs * 3e6) / workPerRow, 8, maxRows);
    }

    /// <summary>
    /// Tensor-core flash attention. Multi-sequence: fa blocks never straddle sequences (grid z = runs); for short runs
    /// (batched decode) the per-query kernels win.
    /// </summary>
    private KernelVariant FlashVariant(AttentionPlan plan)
    {
        var q = plan.Q;
        var runs = plan.Runs;
        var kernel = Lib("attention.slang", "fa", plan.Macros);
        if (runs is null)
        {
            return new KernelVariant(FlashWhen(plan), kernel, b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), 32)));
        }

        return new KernelVariant(FlashWhen(plan), kernel, b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), 32), (uint)Rows(b, runs)));
    }

    /// <summary>Bindings for which the flash variant is chosen (when the device supports it).</summary>
    private static Func<Bindings, bool> FlashWhen(AttentionPlan plan)
    {
        var q = plan.Q;
        var runs = plan.Runs;
        if (runs is null)
        {
            return b => Rows(b, q) > SplitMaxRows;
        }

        return b => DiagnosticSwitches.AttentionSequenceFlash && Rows(b, q) > SplitMaxRows && Rows(b, q) >= 8 * Rows(b, runs);
    }

    /// <summary>
    /// Split-K flash decoding (<c>decode</c> in attention.slang): one workgroup per (GQA group, query row, key range), so
    /// K/V rows are read once per group; the number of key ranges follows each row's context length at run time (short
    /// contexts run one range and skip the combine). Needs subgroup shuffles/reductions and head sizes that are multiples
    /// of 8; returns null when it does not apply (the per-head kernels are the fallback).
    /// </summary>
    private DecodeAttentionPlan? DecodeAttention(AttentionPlan plan, bool flash, long sliceRows)
    {
        const long threads = 128;
        int sg = _opt.SubgroupSize;
        if (plan.Full || !DiagnosticSwitches.AttentionDecode || !_opt.Has(DeviceFeatures.SubgroupOps) || sg < 1 || sg > threads || threads % sg != 0
            || plan.HQ % plan.HKV != 0 || plan.D % 8 != 0 || plan.DV % 8 != 0 || plan.D > 512 || plan.DV > 512)
        {
            return null;
        }

        long group = plan.HQ / plan.HKV;
        long headsPerGroup = Math.Min(group, 8);
        while (group % headsPerGroup != 0)
        {
            headsPerGroup--;
        }

        // KLANES lanes own one key row and the matching value row (8 dims per lane and 16-byte load); each lane group
        // takes DEC_R rows per batch, about 8 loads per lane in flight.
        long keyLanes = Math.Min((long)BitOperations.RoundUpToPowerOf2((ulong)Math.Max(plan.D, plan.DV) / 8), sg);
        long keySlices = (plan.D / 8 + keyLanes - 1) / keyLanes;
        long valueSlices = (plan.DV / 8 + keyLanes - 1) / keyLanes;
        int[] cfg = DiagnosticSwitches.AttentionDecodeConfig;
        long rowsPerBatch = cfg.Length > 3 ? cfg[3] : Math.Max(1, 8 / (keySlices + valueSlices));
        long batch = rowsPerBatch * threads / keyLanes;
        // Fewer than 8 workgroups per key range (few KV heads, e.g. Gemma-3-1B's single one): shorter ranges, so short
        // (sliding-window) contexts still spread over enough workgroups.
        long groups = plan.HKV * (group / headsPerGroup);
        long minKeysTarget = Math.Clamp(cfg[1] * groups / 8, 1, cfg[1]);
        long minKeys = Math.Max(batch, (minKeysTarget + batch - 1) / batch * batch);
        long subgroups = threads / sg;
        long sharedBytes = 4 * subgroups * headsPerGroup * (2 + valueSlices * keyLanes * 8);
        if (sharedBytes > _opt.MaxSharedMemory)
        {
            return null;
        }

        var q = plan.Q;
        long maxRows = plan.Seq is null ? SplitMaxRows : Math.Min(DecodeMaxSequenceRows, sliceRows);
        var flashWhen = flash ? FlashWhen(plan) : null;
        var macros = new Dictionary<string, string>(plan.Macros);
        (string Name, long Value)[] decodeMacros =
        [
            ("HPW", headsPerGroup), ("KLANES", keyLanes), ("DEC_WG", threads), ("DEC_R", rowsPerBatch), ("DEC_BC", batch),
            ("DEC_MAX_SPLITS", cfg[2]), ("DEC_MIN_KEYS", minKeys), ("DEC_TARGET_WG", cfg[0]),
        ];
        foreach (var (name, value) in decodeMacros)
        {
            macros[name] = value.ToString(CultureInfo.InvariantCulture);
        }

        return new DecodeAttentionPlan
        {
            Macros = macros,
            Groups = groups,
            MaxRows = maxRows,
            TargetWorkgroups = cfg[0],
            MaxSplitCount = cfg[2],
            When = b => Rows(b, q) <= maxRows && (flashWhen is null || !flashWhen(b)),
        };
    }

    /// <summary>Scratch for the partials of split-K decoding (only rows that run more than one key range use it).</summary>
    private TensorRef DecodeScratch(AttentionPlan plan, DecodeAttentionPlan decode, long maxRows)
    {
        long partials = 1;
        for (long rows = 1; rows <= Math.Min(maxRows, decode.MaxRows); rows++)
        {
            long splits = decode.MaxSplits(rows);
            if (splits > 1)
            {
                partials = Math.Max(partials, splits * rows);
            }
        }

        long bytes = partials * plan.HQ * (plan.DV + 4) * 4;   // DEC_PS floats per partial
        _attnScratch ??= NewBuffer(BufferKind.Activations, "attn-split", 0);
        _attnScratch.Bytes = Math.Max(_attnScratch.Bytes, bytes);
        return new TensorRef(_attnScratch, 0, DataType.F32, [bytes / 4], [1], null);
    }

    private void EmitAttentionLaunch(AttentionPlan plan, List<KernelVariant> variants, TensorRef part, long firstRow, bool split)
    {
        var o = plan.Output;
        var sinks = plan.Sinks;
        var seq = plan.Seq;
        var spans = plan.Spans;
        Emit($"attention {plan.Node.Output.Name}", [plan.Node],
            variants,
            [Pointer(plan.Q), Pointer(plan.K), Pointer(plan.V), Pointer(o), Pointer(part), Pointer(sinks ?? o), new PointerArg(_runtime, 0),
             .. SeqArgs(seq, plan.Runs, spans), TokenArg(plan.Q), new UIntArg((uint)firstRow)],
            [plan.Q, plan.K, plan.V, .. sinks is null ? [] : (TensorRef[])[sinks], .. seq is null ? [] : (TensorRef[])[seq, plan.Runs!],
             .. spans is null ? [] : (TensorRef[])[spans]],
            split ? [o, part] : [o]);
    }

    /// <summary>Merges the per-split partial results of decode attention into the output.</summary>
    private void EmitAttentionCombine(AttentionPlan plan, TensorRef part, Dictionary<string, string> splitMacros)
    {
        var q = plan.Q;
        var o = plan.Output;
        var sinks = plan.Sinks;
        Emit($"attention {plan.Node.Output.Name} combine", [plan.Node],
            [new KernelVariant(b => Rows(b, q) <= SplitMaxRows, Lib("attention.slang", "combine", splitMacros), b => new Grid((uint)plan.HQ, (uint)Rows(b, q)))],
            [Pointer(part), Pointer(o), Pointer(sinks ?? o), TokenArg(q)],
            sinks is null ? [part] : [part, sinks], [o]);
    }

    /// <summary>Merges the partials of the rows that split-K decoding ran in more than one key range.</summary>
    private void EmitDecodeCombine(AttentionPlan plan, TensorRef part, DecodeAttentionPlan decode)
    {
        var q = plan.Q;
        var o = plan.Output;
        var sinks = plan.Sinks;
        var seq = plan.Seq;
        Emit($"attention {plan.Node.Output.Name} combine", [plan.Node],
            [new KernelVariant(decode.When, Lib("attention.slang", "decode_combine", decode.Macros), b => new Grid((uint)plan.HQ, (uint)Rows(b, q)))],
            [Pointer(part), Pointer(o), Pointer(sinks ?? o), new PointerArg(_runtime, 0), .. SeqArgs(seq), TokenArg(q)],
            [part, .. sinks is null ? [] : (TensorRef[])[sinks], .. seq is null ? [] : (TensorRef[])[seq]], [o]);
    }
}
