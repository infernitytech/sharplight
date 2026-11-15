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

    /// <summary>Workgroups that key-split flash attention aims for (a few per compute unit on large GPUs).</summary>
    private const long FlashTargetWorkgroups = 768;

    /// <summary>Multi-sequence query rows (batched decode) up to which split-K decoding replaces the per-head kernel.</summary>
    private const long DecodeMaxSequenceRows = 64;

    /// <summary>
    /// Mixed multi-sequence steps (decode rows next to prompt chunks): runs of at least this many rows go to flash attention,
    /// shorter ones (decode rows, speculative verification rows) to split-K decoding.
    /// </summary>
    private const long MixedFlashMinRunRows = 16;

    private BufferDecl? _attnScratch;

    /// <summary>Index of the runtime scalar holding the cache position of a stateful node.</summary>
    private int Slot(Node n) => _scalarSlots[n.Attributes.Get("offset", WellKnown.PastScalar)];

    /// <summary>Multi-sequence ops: cache rows per sequence slot (cache rows / attribute <c>slots</c>; 0 for a paged cache).</summary>
    private static long SlotContext(Node n, TensorRef cache)
    {
        if (n.Attributes.Get("block_size", 0L) > 0)
        {
            return 0;
        }

        long slots = n.Attributes.Get("slots", 1L);
        if (slots < 1 || cache.Shape[0] % slots != 0)
        {
            throw new NotSupportedException($"{n.Op.Name}: cache rows {cache.Shape[0]} not divisible by {slots} slots.");
        }

        return cache.Shape[0] / slots;
    }

    private TensorRef DenseInput(Value v, string name) => EnsureDense(Ref(v), name);

    /// <summary>
    /// The block table [slots, blocks per slot] of a paged multi-sequence op (input <paramref name="index"/>, attribute
    /// <c>block_size</c>), or null for a contiguous cache.
    /// </summary>
    private TensorRef? PageTable(Node n, int index)
        => n.Attributes.Get("block_size", 0L) > 0 && n.Inputs.Length > index ? DenseInput(n.Inputs[index], "kv-pages") : null;

    /// <summary>The PAGED / PAGE_BS / PAGE_MB kernel macros of a paged op (see common.slang).</summary>
    private static (string, object)[] PageMacros(Node n, TensorRef? pages)
        => [("PAGED", pages is not null), ("PAGE_BS", pages is null ? 1 : n.Attributes.Get("block_size", 0L)), ("PAGE_MB", pages is null ? 1 : pages.Shape[1])];

    /// <summary>Whether a KV-cache type is one of the quantized block formats (q8_0, q4_0) the attention kernels read.</summary>
    internal static bool IsQuantizedKv(DataType type) => type == DataType.Q8_0 || type == DataType.Q4_0;

    /// <summary>The <c>KV_FMT</c> / <c>DST_FMT</c> kernel id (the <c>FMT_*</c> ids of common.slang) of a KV-cache type.</summary>
    private static int KvFormat(DataType type) => s_weightFormats.TryGetValue(type, out int id) && (type == DataType.F32 || type == DataType.F16 || IsQuantizedKv(type))
        ? id
        : throw new NotSupportedException($"KV cache type {type}");

    private static KernelArg[] SeqArgs(params TensorRef?[] tables) => [.. tables.OfType<TensorRef>().Select(t => (KernelArg)Pointer(t))];

    /// <summary>The present tables of <paramref name="tables"/> (kernel inputs to track).</summary>
    private static TensorRef[] PresentTables(params TensorRef?[] tables) => [.. tables.OfType<TensorRef>()];

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
        bool quantized = IsQuantizedKv(state.Type);
        if (!f16 && !quantized && state.Type != DataType.F32)
        {
            throw new NotSupportedException($"Cache type {state.Type}");
        }

        var seq = n.Inputs.Length > 2 ? DenseInput(n.Inputs[2], "seq-rows") : null;
        var pages = PageTable(n, 3);
        long compression = n.Attributes.Get("compression", 1L);
        if (compression > 1 && quantized)
        {
            throw new NotSupportedException("A compressed cache must be f32 or f16.");
        }

        var macros = Macros([("R1", r1), ("INNER", inner), ("S0", s0), ("S1", s1), ("DST_F16", f16), ("RT_N_PAST", seq is null ? Slot(n) : 0),
            ("MULTI_SEQ", seq is not null), ("SLOT_CTX", seq is null || compression > 1 ? 0 : SlotContext(n, state)), .. PageMacros(n, pages),
            ("KV_COMP", compression), ("COMP_SLOT_CTX", seq is null || compression == 1 ? 0 : SlotContext(n, state))]);
        macros["DST_FMT"] = KvFormat(state.Type).ToString(CultureInfo.InvariantCulture);
        var k = Lib("data.slang", quantized ? "cache_write_q" : "cache_write", macros);
        long perThread = quantized ? 32 : 1;   // the quantizing kernel writes one 32-value block per thread
        var tx = x;
        Emit($"cache_write {state.Buffer.Name}", [n],
            [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * r1 * inner / perThread, 128)))],
            [Pointer(x), Pointer(state), new PointerArg(_runtime, 0), .. SeqArgs(seq, pages), TokenArg(x)],
            [x, .. PresentTables(seq, pages)], [state]);
    }

    private void LowerAttention(Node n)
    {
        var plan = PlanAttention(n);
        if (plan.Selection is not null || plan.Compressed is not null)
        {
            EmitSelectedAttention(plan);
            return;
        }

        if (MqaAttention(plan) is { } mqa)
        {
            EmitMqaAttention(plan, mqa);
            return;
        }

        var q = plan.Q;
        var o = plan.Output;

        bool flash = CanUseFlashAttention(plan);
        bool simtFlash = !flash && SimtEncoderFlash(plan);
        long maxRows = q.Symbol is null ? q.Shape[0] : Bound(Dim.Sym(q.Symbol));
        long sliceRows = SliceRows(plan, flash || simtFlash, maxRows);
        int slices = (int)((maxRows + sliceRows - 1) / sliceRows);

        // Few queries (decode): one workgroup per (head, query) leaves most of the GPU idle, so the keys are split
        // across several workgroups (flash decoding) and merged by a small combine kernel. Preferred: the GQA-grouped
        // split-K kernel with a run-time split count; otherwise SPLITS fixed key ranges per head.
        var decode = DecodeAttention(plan, flash, sliceRows);
        int splits = (int)Math.Clamp((256 + plan.HQ - 1) / plan.HQ, 1, 32);
        bool split = decode is null && !plan.Full && splits > 1 && DiagnosticSwitches.AttentionSplit;
        int keySplits = (flash || simtFlash) && slices == 1 ? FlashKeySplits(plan, maxRows) : 1;
        var part = decode is not null ? DecodeScratch(plan, decode, maxRows) : split ? SplitScratch(plan, splits)
            : keySplits > 1 ? AttentionScratch(keySplits * maxRows * plan.HQ * (plan.DV + 2) * 4) : o;
        var splitMacros = new Dictionary<string, string>(plan.Macros) { ["SPLITS"] = splits.ToString(CultureInfo.InvariantCulture) };

        // Steps of many sequences beyond the decode kernel's rows that are not long runs only (decode rows next to prompt
        // chunks): split-K decoding for the short runs, then flash attention for the long ones.
        var mixed = flash && slices == 1 && decode is not null ? MixedAttention(plan, decode) : null;
        if (mixed is not null)
        {
            EmitAttentionLaunch(plan, [mixed.Value.Decode], part, 0, true);
        }

        var scalarKernel = Lib("attention.slang", "main", plan.Macros);
        for (int slice = 0; slice < slices; slice++)
        {
            long firstRow = slice * sliceRows;
            var variants = new List<KernelVariant>();
            if (mixed is not null)
            {
                variants.Add(mixed.Value.Flash);
            }

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
                variants.Add(FlashVariant(plan, keySplits));
            }

            if (slice == 0 && simtFlash)
            {
                variants.Add(SimtFlashVariant(plan, keySplits));
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

            EmitAttentionLaunch(plan, variants, part, firstRow, split || decode is not null || keySplits > 1);
        }

        if (keySplits > 1)
        {
            EmitFlashCombine(plan, part, keySplits);
        }
        else if (decode is not null)
        {
            EmitDecodeCombine(plan, part, decode);
        }
        else if (split)
        {
            EmitAttentionCombine(plan, part, splitMacros);
        }
    }

    /// <summary>
    /// Resolves the operands of an Attention / CachedAttention node and builds the kernel macros. Bidirectional attention reads
    /// f32 K / V with wider row strides (e.g. slices of a fused Q / K / V projection) in place when the kernels that serve it
    /// (the encoder flash kernel) takes strided rows; otherwise they are copied to dense rows.
    /// </summary>
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

        bool stridedKv = full && DiagnosticSwitches.AttentionStridedKv && StridedKvReadable(k) && StridedKvReadable(v);
        if (full && !stridedKv)
        {
            k = AsDenseF32(k, "attn-k");
            v = AsDenseF32(v, "attn-v");
        }

        var output = Produce(n.Output);
        bool hasSinks = !full && n.Attributes.Get("sinks", false);
        bool hasSelection = !full && n.Attributes.Get("selection", false);
        long compression = full ? 1 : n.Attributes.Get("compression", 1L);
        bool hasCompressed = !full && n.Attributes.Has("compression");
        // inputs: q, k, v, [sinks], [compressed cache], [selection], [rows, runs, [block table]]
        int sinksInputs = (hasSinks ? 1 : 0) + (hasCompressed ? 1 : 0) + (hasSelection ? 1 : 0);
        TensorRef? sinks = hasSinks ? Ref(n.Inputs[3]) : null;   // attention sinks [HQ] f32
        TensorRef? compressed = hasCompressed ? Ref(n.Inputs[hasSinks ? 4 : 3]) : null;   // compressed cache [entries, HKV, VRS]
        TensorRef? selection = hasSelection ? DenseInput(n.Inputs[2 + sinksInputs], "attn-selection") : null;   // sparse attention: [T, K] or [T, Hs, K] key positions
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
        TensorRef? mask = null;
        TensorRef? pages = null;
        bool causal = full && n.Attributes.Get("causal", false);
        if (full && n.Inputs.Length > 3)
        {
            if (n.Inputs[3].Type == DataType.I32)
            {
                spans = DenseInput(n.Inputs[3], "attn-spans");
            }
            else
            {
                mask = DenseInput(n.Inputs[3], "attn-mask");
            }
        }

        // Batched form: equal-length sequences attend within themselves (query rows / SEQ_ROWS sequences).
        long sequenceRows = full ? BatchedAttention.SequenceRows(n) : 0;
        long sequenceKeys = sequenceRows > 0 ? BatchedAttention.SequenceKeys(n) : 0;

        // Cross-attention (keys of another sequence, e.g. an encoder's output): the kernels take the key count as a parameter.
        bool cross = full && spans is null && sequenceRows == 0 && n.Inputs[0].Shape[0] != n.Inputs[1].Shape[0];

        if (!full && n.Inputs.Length > 3 + sinksInputs)
        {
            seq = DenseInput(n.Inputs[3 + sinksInputs], "seq-rows");
            runs = n.Inputs.Length > 4 + sinksInputs
                ? DenseInput(n.Inputs[4 + sinksInputs], "seq-runs")
                : throw new NotSupportedException("Multi-sequence attention needs the runs table.");
            pages = PageTable(n, 5 + sinksInputs);
        }

        var macros = Macros(
            [.. PageMacros(n, pages),
            ("MULTI_SEQ", seq is not null), ("SLOT_CTX", seq is null ? 0 : SlotContext(n, k)), ("HQ", HQ), ("HKV", HKV), ("D", D), ("DV", DV), ("VRS", valueRowStride),
            ("QS0", q.Strides[0]), ("QS1", q.Strides[1]), ("SCALE", n.Attributes.Get("scale", 1f)), ("KV_F16", k.Type == DataType.F16), ("KV_FMT", KvFormat(k.Type)),
            ("FULL_ATTN", full), ("RT_N_PAST", full || seq is not null ? 0 : Slot(n)), ("WINDOW", window), ("CHUNK", n.Attributes.Get("chunk", 0)),
            ("HAS_SOFTCAP", softcap > 0), ("SOFTCAP", softcap > 0 ? softcap : 1f), ("HAS_SINKS", sinks is not null),
            ("SPANS", spans is not null), ("CAUSAL", causal), ("CROSS_ATTN", cross), ("SELECTED", selection?.Shape[^1] ?? 0), ("SEL_HEADS", selection is { Shape.Length: 3 } ? selection.Shape[1] : 1),
            ("SEQ_ROWS", sequenceRows), ("SEQ_KEYS", sequenceKeys), ("HAS_MASK", mask is not null), ("MASK_PER_ROW", mask is not null && n.Inputs[3].Rank == 2),
            .. AlibiMacros(n),
            ("COMP_ATTN", compressed is not null), ("KV_COMP", compression), ("COMP_SLOT_CTX", seq is null || compressed is null ? 0 : SlotContext(n, compressed))]);
        if (compressed is not null && compressed.Type != k.Type)
        {
            throw new NotSupportedException("The compressed cache must have the key cache's type.");
        }

        if (stridedKv)
        {
            macros["KV_KS0"] = k.Strides[0].ToString(CultureInfo.InvariantCulture);
            macros["KV_KS1"] = k.Strides[1].ToString(CultureInfo.InvariantCulture);
            macros["KV_VS0"] = v.Strides[0].ToString(CultureInfo.InvariantCulture);
            macros["KV_VS1"] = v.Strides[1].ToString(CultureInfo.InvariantCulture);
        }

        AttentionPlan Build() => new AttentionPlan
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
            Pages = pages,
            Spans = spans,
            Mask = mask,
            SequenceRows = sequenceRows,
            Selection = selection,
            Compressed = compressed,
            Cross = cross,
            HQ = HQ,
            HKV = HKV,
            D = D,
            DV = DV,
            Macros = macros,
        };
        var plan = Build();
        if (stridedKv && (!CanUseFlashAttention(plan) || !EncoderFlash(plan) || MqaAttention(plan) is not null))
        {
            // Another kernel serves this attention: dense K / V rows (the plan's other operands are unchanged).
            k = AsDenseF32(k, "attn-k");
            v = AsDenseF32(v, "attn-v");
            foreach (var key in (string[])["KV_KS0", "KV_KS1", "KV_VS0", "KV_VS1"])
            {
                macros.Remove(key);
            }

            plan = Build();
        }

        return plan;
    }

    /// <summary>
    /// True when the strided-K/V kernels can read <paramref name="t"/> in place: f32 [T, H, D] with unit element stride, row
    /// and head strides in 8-value groups and 16-byte alignment.
    /// </summary>
    private static bool StridedKvReadable(TensorRef t)
        => t.Rank == 3 && t.Type == DataType.F32 && t.Strides[2] == 1 && t.Strides[1] % 8 == 0 && t.Strides[0] % 8 == 0 && t.ByteOffset % 16 == 0
           && t.Shape[2] % 8 == 0;

    /// <summary>
    /// The HAS_ALIBI / ALIBI_SLOPES and HAS_POS_BIAS / POS_BIAS_RANGE / POS_BIAS_TABLE kernel macros of an attention node: its
    /// <c>alibi_slopes</c> and <c>position_bias</c> become constant arrays of the kernel (round-trip float literals), so they need no buffer
    /// and the kernels no extra binding.
    /// </summary>
    private static (string, object)[] AlibiMacros(Node n)
    {
        (string, object)[] positionBias = n.Attributes.Get<float[]?>("position_bias", null) is { } table
            ? [("HAS_POS_BIAS", true), ("POS_BIAS_RANGE", n.Attributes.Get("position_bias_range", 0)), ("POS_BIAS_TABLE", Literals(table))]
            : [("HAS_POS_BIAS", false)];
        if (n.Attributes.Get<float[]?>("alibi_slopes", null) is not { } slopes)
        {
            return [("HAS_ALIBI", false), .. positionBias];
        }

        return [("HAS_ALIBI", true), ("ALIBI_SLOPES", Literals(slopes)), .. positionBias];
    }

    /// <summary>A Slang initializer list of round-trip float literals.</summary>
    private static string Literals(float[] values) => "{ " + string.Join(", ", values.Select(s => s.ToString("R", CultureInfo.InvariantCulture) + "f")) + " }";

    /// <summary>The shared f32 scratch buffer of split attention, grown to at least <paramref name="bytes"/>.</summary>
    private TensorRef AttentionScratch(long bytes)
    {
        _attnScratch ??= NewBuffer(BufferKind.Activations, "attn-split", 0);
        _attnScratch.Bytes = Math.Max(_attnScratch.Bytes, bytes);
        return new TensorRef(_attnScratch, 0, DataType.F32, [bytes / 4], [1], null);
    }

    /// <summary>Scratch for the partial results of split-key decode attention: (max, sum, accumulator) per split.</summary>
    private TensorRef SplitScratch(AttentionPlan plan, int splits)
    {
        long rowsMax = plan.Q.Shape[0];
        long bytes = splits * Math.Min(rowsMax, SplitMaxRows) * plan.HQ * (plan.DV + 2) * 4;
        return AttentionScratch(bytes);
    }

    /// <summary>
    /// Many queries (prefill, vision encoders): tensor-core flash attention (f16 Q/K/V/P) when the device and shapes allow
    /// it. Head sizes are padded in the kernel; the padded tiles must fit shared memory. f32 caches keep the exact kernel
    /// unless reduced precision is allowed; quantized caches are converted to f16 tiles.
    /// </summary>
    private bool CanUseFlashAttention(AttentionPlan plan)
    {
        long paddedD = (plan.D + 15) / 16 * 16;
        long paddedDV = (plan.DV + 63) / 64 * 64;
        long paddedKV = Math.Max(paddedD, paddedDV);
        long halfTiles = 32 * (paddedD + 8) * 2 + 32 * (paddedKV + 8) * 2 + 32 * 40 * 2;
        long sharedBytes = halfTiles * (SplitFlash(plan) ? 2 : 1) + 32 * 32 * 4 + 32 * 64 * 4;
        bool typeAllowed = plan.K.Type == DataType.F16 || IsQuantizedKv(plan.K.Type)
                           || (plan.K.Type == DataType.F32 && (_opt.Compile.AllowReducedPrecision || SplitFlash(plan)));
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
        return Math.Clamp((long)(_opt.SubmissionBudgetMs * 3e6) / workPerRow, Math.Min(8, maxRows), maxRows);
    }

    /// <summary>
    /// Tensor-core flash attention. Multi-sequence: fa blocks never straddle sequences (grid z = runs); for short runs
    /// (batched decode) the per-query kernels win.
    /// </summary>
    private KernelVariant FlashVariant(AttentionPlan plan, int keySplits)
    {
        var q = plan.Q;
        var runs = plan.Runs;
        var macros = new Dictionary<string, string>(plan.Macros) { ["FA_SPLIT"] = SplitFlash(plan) ? "1" : "0", ["FA_KSPLIT"] = keySplits.ToString(CultureInfo.InvariantCulture) };
        bool encoder = EncoderFlash(plan);
        var kernel = Lib("attention.slang", encoder ? "fe" : "fa", macros);
        uint blockRows = encoder ? 64u : 32u;
        if (plan.SequenceRows > 0)
        {
            long sequenceRows = plan.SequenceRows;
            return new KernelVariant(FlashWhen(plan), kernel,
                b => new Grid((uint)plan.HQ, Grid.Div(sequenceRows, blockRows), (uint)(Rows(b, q) / sequenceRows * keySplits)));
        }

        if (runs is null)
        {
            return new KernelVariant(FlashWhen(plan), kernel, b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), blockRows), (uint)keySplits));
        }

        return new KernelVariant(FlashWhen(plan), kernel, b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), 32), (uint)Rows(b, runs)));
    }

    /// <summary>
    /// Whether flash attention runs the encoder kernel <c>fe</c> (64 queries per workgroup, each subgroup keeping its query
    /// fragments and output accumulators in registers): plain bidirectional attention over f32 or f16 K/V (an optional
    /// additive mask, batched or cross form) with padded head sizes up to 128, unless <c>SHARPLIGHT_ATTN_FE=0</c>.
    /// </summary>
    private bool EncoderFlash(AttentionPlan plan)
    {
        var a = plan.Node.Attributes;
        bool plain = plan.Full && plan.Spans is null && plan.Sinks is null && plan.Runs is null && plan.Selection is null
                     && a.Get("window", 0) == 0 && !a.Get("causal", false) && a.Get("softcap", 0f) <= 0
                     && a.Get<float[]?>("alibi_slopes", null) is null && a.Get<float[]?>("position_bias", null) is null;
        bool kv = (plan.K.Type == DataType.F32 || plan.K.Type == DataType.F16) && plan.V.Type == plan.K.Type;
        long paddedD = (plan.D + 15) / 16 * 16;
        long paddedDV = (plan.DV + 15) / 16 * 16;
        long kvHalves = 2 * 32 * (paddedD + 8) + 2 * 32 * (paddedDV + 8);
        long buffer = Math.Max(kvHalves, 2 * 64 * (paddedD + 8)) * 2;
        long sharedBytes = buffer + 4 * 16 * 36 * 4 + 4 * 16 * 40 * 2 * (SplitFlash(plan) ? 2 : 1) + 4 * 256 * 4;
        return DiagnosticSwitches.AttentionEncoderFlash && plain && kv && paddedD <= 128 && paddedDV <= 128
               && sharedBytes <= _opt.MaxSharedMemory;
    }

    /// <summary>
    /// Whether bidirectional attention on a device without the tensor-core flash kernels runs the f32 encoder flash kernel
    /// <c>fes</c> (64 queries per workgroup, one per thread, K/V tiles shared): the forms <c>fe</c> serves (see
    /// <see cref="EncoderFlash"/>) with head sizes that are multiples of 8 up to 64, unless <c>SHARPLIGHT_ATTN_FES=0</c>.
    /// The per-query kernel re-reads every key for each query (vit-base on an integrated GPU: 16.5 ms per layer).
    /// </summary>
    private bool SimtEncoderFlash(AttentionPlan plan)
    {
        var a = plan.Node.Attributes;
        bool plain = plan.Full && plan.Spans is null && plan.Sinks is null && plan.Runs is null && plan.Selection is null
                     && a.Get("window", 0) == 0 && !a.Get("causal", false) && a.Get("softcap", 0f) <= 0
                     && a.Get<float[]?>("alibi_slopes", null) is null && a.Get<float[]?>("position_bias", null) is null;
        bool kv = (plan.K.Type == DataType.F32 || plan.K.Type == DataType.F16) && plan.V.Type == plan.K.Type;
        long sharedBytes = 32 * (plan.D + plan.DV) * 4;
        return DiagnosticSwitches.AttentionSimtFlash && plain && kv && plan.HQ % plan.HKV == 0 && plan.D % 8 == 0 && plan.DV % 8 == 0
               && plan.D <= 64 && plan.DV <= 64 && sharedBytes <= _opt.MaxSharedMemory;
    }

    /// <summary>The <c>fes</c> variant (see <see cref="SimtEncoderFlash"/>): grid (heads, 64-query blocks, sequences x key ranges).</summary>
    private KernelVariant SimtFlashVariant(AttentionPlan plan, int keySplits)
    {
        var q = plan.Q;
        var macros = new Dictionary<string, string>(plan.Macros) { ["FA_KSPLIT"] = keySplits.ToString(CultureInfo.InvariantCulture) };
        var kernel = Lib("attention.slang", "fes", macros);
        if (plan.SequenceRows > 0)
        {
            long sequenceRows = plan.SequenceRows;
            return new KernelVariant(FlashWhen(plan), kernel,
                b => new Grid((uint)plan.HQ, Grid.Div(sequenceRows, 64), (uint)(Rows(b, q) / sequenceRows * keySplits)));
        }

        return new KernelVariant(FlashWhen(plan), kernel, b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), 64), (uint)keySplits));
    }

    /// <summary>
    /// Whether flash attention over f32 K/V runs in the split-f16 form (FA_SPLIT: hi·hi + hi·lo + lo·hi products, near-f32
    /// accuracy): under strict precision, unless disabled with <c>SHARPLIGHT_ATTN_SPLIT_F16=0</c>. Reduced precision uses
    /// single f16 products.
    /// </summary>
    private bool SplitFlash(AttentionPlan plan)
        => plan.K.Type == DataType.F32 && !_opt.Compile.AllowReducedPrecision && DiagnosticSwitches.AttentionSplitHalf;

    /// <summary>
    /// Key ranges per query block of bidirectional flash attention (FA_KSPLIT): enough workgroups for about
    /// <see cref="FlashTargetWorkgroups"/>, at least two key tiles per range, at most 8 ranges; 1 when the attention has spans,
    /// a window, causal masking or sinks, or when <c>SHARPLIGHT_ATTN_FA_KSPLIT=0</c>.
    /// </summary>
    private int FlashKeySplits(AttentionPlan plan, long maxRows)
    {
        var a = plan.Node.Attributes;
        if (!plan.Full || plan.Spans is not null || plan.Sinks is not null || a.Get("window", 0) > 0 || a.Get("causal", false)
            || !DiagnosticSwitches.AttentionFlashKeySplit)
        {
            return 1;
        }

        long sequences = plan.SequenceRows > 0 ? Math.Max(1, maxRows / plan.SequenceRows) : 1;
        long rowsPerSequence = plan.SequenceRows > 0 ? plan.SequenceRows : maxRows;
        long keys = plan.SequenceRows > 0 ? BatchedAttention.SequenceKeys(plan.Node) : plan.K.Symbol is null ? plan.K.Shape[0] : Bound(Dim.Sym(plan.K.Symbol));
        long blockRows = EncoderFlash(plan) || !CanUseFlashAttention(plan) ? 64 : 32;
        long workgroups = plan.HQ * ((rowsPerSequence + blockRows - 1) / blockRows) * sequences;
        long wanted = (FlashTargetWorkgroups + workgroups - 1) / workgroups;
        return (int)Math.Clamp(Math.Min(wanted, (keys + 31) / 32 / 2), 1, 8);
    }

    /// <summary>Merges the key-range partials of split flash attention into the output.</summary>
    private void EmitFlashCombine(AttentionPlan plan, TensorRef part, int keySplits)
    {
        var q = plan.Q;
        var o = plan.Output;
        var macros = new Dictionary<string, string>(plan.Macros) { ["SPLITS"] = keySplits.ToString(CultureInfo.InvariantCulture) };
        Emit($"attention {plan.Node.Output.Name} combine", [plan.Node],
            [new KernelVariant(FlashWhen(plan), Lib("attention.slang", "combine", macros), b => new Grid((uint)plan.HQ, (uint)Rows(b, q)))],
            [Pointer(part), Pointer(o), Pointer(o), TokenArg(q)],
            [part], [o]);
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

        if (InvariantRows > 1)
        {
            macros["DEC_INVARIANT_ROWS"] = InvariantRows.ToString(CultureInfo.InvariantCulture);
        }

        return new DecodeAttentionPlan
        {
            Macros = macros,
            Groups = groups,
            MaxRows = maxRows,
            TargetWorkgroups = cfg[0],
            MaxSplitCount = cfg[2],
            InvariantRows = InvariantRows,
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
        return AttentionScratch(bytes);
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
             .. SeqArgs(seq, plan.Runs, plan.Pages, spans, plan.Selection, plan.Compressed, plan.Mask), TokenArg(plan.Q), new UIntArg((uint)firstRow), .. plan.Cross ? (KernelArg[])[TokenArg(plan.K)] : []],
            [plan.Q, plan.K, plan.V, .. sinks is null ? [] : (TensorRef[])[sinks], .. PresentTables(seq, plan.Runs, plan.Pages, spans, plan.Selection, plan.Compressed, plan.Mask)],
            split ? [o, part] : [o]);
    }

    /// <summary>
    /// Sparse attention (a selection input): the per-(head, query) kernel visits only the selected key positions, so its cost follows the selection
    /// width instead of the context. Compressed attention (a window of keys plus compressed entries) runs on the same kernel. Long batches are sliced like the scalar kernel's on devices with a submission budget.
    /// </summary>
    private void EmitSelectedAttention(AttentionPlan plan)
    {
        var q = plan.Q;
        long maxRows = q.Symbol is null ? q.Shape[0] : Bound(Dim.Sym(q.Symbol));
        long sliceRows = maxRows;
        if (_opt.SubmissionBudgetMs > 0)
        {
            long entries = plan.Selection?.Shape[^1] ?? 0;
            long keys = plan.Compressed is null ? entries : entries + plan.Node.Attributes.Get("window", 0)
                + (plan.Selection is null ? plan.Compressed.Shape[0] : 0);
            long workPerRow = Math.Max(1, plan.HQ * Math.Max(1, keys) * (plan.D + plan.DV));
            sliceRows = Math.Clamp((long)(_opt.SubmissionBudgetMs * 3e6) / workPerRow, Math.Min(8, maxRows), maxRows);
        }

        var kernel = Lib("attention.slang", "main", plan.Macros);
        for (long firstRow = 0; firstRow < maxRows; firstRow += sliceRows)
        {
            long first = firstRow;
            var variant = new KernelVariant(b => Rows(b, q) > first, kernel, b => new Grid((uint)plan.HQ, (uint)Math.Min(sliceRows, Rows(b, q) - first)));
            EmitAttentionLaunch(plan, [variant], plan.Output, first, split: false);
        }
    }

    /// <summary>Merges the per-split partial results of the fixed-split attention kernel into the output.</summary>
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

    /// <summary>
    /// The variants of a mixed multi-sequence step (see <see cref="MixedFlashMinRunRows"/>), or null when the attention has no
    /// run table or mixing is switched off (<c>SHARPLIGHT_ATTN_MIXED=0</c>). They apply above the decode kernel's row limit
    /// when a step has more than one run but too many runs for flash attention over all of them (fewer than 8 rows per run):
    /// decode rows next to prompt chunks. A step whose runs are all short or all long just skips the other kernel's
    /// workgroups. Beyond <see cref="DecodeAttentionPlan.MaxRows"/> rows the decode kernel runs one key range per row
    /// (<c>DEC_SPLIT_ROWS</c>) and writes its output directly, so no combine is needed.
    /// </summary>
    private (KernelVariant Decode, KernelVariant Flash)? MixedAttention(AttentionPlan plan, DecodeAttentionPlan decode)
    {
        var q = plan.Q;
        var runs = plan.Runs;
        if (runs is null || !DiagnosticSwitches.AttentionMixed)
        {
            return null;
        }

        var flashAll = FlashWhen(plan);
        Func<Bindings, bool> when = b => Rows(b, q) > decode.MaxRows && Rows(b, runs) > 1 && !flashAll(b);
        var decodeMacros = new Dictionary<string, string>(decode.Macros)
        {
            ["MIX_RUN"] = MixedFlashMinRunRows.ToString(CultureInfo.InvariantCulture),
            ["DEC_SPLIT_ROWS"] = decode.MaxRows.ToString(CultureInfo.InvariantCulture),
        };
        var flashMacros = new Dictionary<string, string>(plan.Macros) { ["MIX_RUN"] = MixedFlashMinRunRows.ToString(CultureInfo.InvariantCulture) };
        return (new KernelVariant(when, Lib("attention.slang", "decode", decodeMacros), b => new Grid((uint)decode.Groups, (uint)Rows(b, q), 1)),
            new KernelVariant(when, Lib("attention.slang", "fa", flashMacros), b => new Grid((uint)plan.HQ, Grid.Div(Rows(b, q), 32), (uint)Rows(b, runs))));
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
