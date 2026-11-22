using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    // ---- normalization -----------------------------------------------------------------------
    private void LowerNorm(Node n, int mode)
    {
        var x = Ref(n.Inputs[0]);
        x = PrepareRows3(x, "norm", out long r1, out long s0, out long s1);
        long D = x.Shape[^1];
        TensorRef? w = n.Inputs.Length > 1 ? Ref(n.Inputs[1]) : null;
        if (w is not null && (w.Type != DataType.F32 || !w.IsDense))
        {
            throw new NotSupportedException("Norm weight must be dense f32.");
        }

        var y = Produce(n.Output);
        float eps = n.Attributes.Get<float>("eps");
        if (D >= 1024)
        {
            // Wide rows (hidden-size norms): a workgroup per row with all loads in flight beats one subgroup per row.
            EmitRowNormKernel($"{n.Op.Name} {n.Output.Name}", [n], x.Shape, x.Symbol, [x], [], "a0", [], [], [(y, "nrm")], w, mode, eps, [n.Output]);
            return;
        }

        var k = Lib("norm.slang", "main", Macros(("D", D), ("R1", r1), ("S0", s0), ("S1", s1), ("EPS", eps), ("MODE", mode), ("HAS_W", w is not null)));
        var tx = x;
        Emit($"{n.Op.Name} {n.Output.Name}", [n],
            [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * r1, 4)))],
            [Pointer(x), w is null ? Pointer(x) : Pointer(w), Pointer(y), TokenArg(x)],
            w is null ? [x] : [x, w], [y]);
    }

    // ---- rope --------------------------------------------------------------------------------
    /// <param name="n">The Rope node.</param>
    /// <param name="cacheWrite">When set, the rotated rows are stored straight into this CacheWrite's cache (no y).</param>
    private void LowerRope(Node n, Node? cacheWrite = null)
    {
        // An absorbed RmsNorm of the input is computed in the kernel.
        var norm = _absorbedNorms.GetValueOrDefault(n.Inputs[0]);
        var x = Ref(norm?.Inputs[0] ?? n.Inputs[0]);
        var pos = Ref(n.Inputs[1]);
        if (x.Rank != 3 || x.Strides[2] != 1)
        {
            x = DenseTemp(x, "rope");
        }

        pos = EnsureDense(pos, "rope-pos");

        TensorRef? normWeight = norm is not null && norm.Inputs.Length > 1 ? Ref(norm.Inputs[1]) : null;
        var y = cacheWrite is null ? Produce(n.Output) : Ref(cacheWrite.Inputs[0]);
        var frequencyFactors = n.Inputs.Length > 2 ? Ref(n.Inputs[2]) : null;   // optional per-frequency divisors (f32 [rot/2])
        var ropeMacros = RopeMacros(n.Attributes, x, pos, frequencyFactors is not null);
        var seq = cacheWrite is not null && cacheWrite.Inputs.Length > 2 ? DenseInput(cacheWrite.Inputs[2], "seq-rows") : null;
        var pages = cacheWrite is null ? null : PageTable(cacheWrite, 3);
        var tx = x;
        long H = x.Shape[1];
        if (norm is not null || cacheWrite is not null)
        {
            if (norm is not null && normWeight is null)
            {
                throw new NotSupportedException("Fused rope norm expects a weight.");
            }

            if (cacheWrite is not null && y.Type != DataType.F16 && y.Type != DataType.F32)
            {
                throw new NotSupportedException($"Cache type {y.Type}");
            }

            (string, object)[] fusedMacros =
            [
                .. ropeMacros,
                ("NORM", norm is not null),
                ("EPS", norm?.Attributes.Get<float>("eps") ?? 0f),
                ("CACHE_OUT", cacheWrite is not null),
                ("CACHE_F16", y.Type == DataType.F16),
                ("RT_N_PAST", cacheWrite is null || seq is not null ? 0 : Slot(cacheWrite)),
                ("MULTI_SEQ", seq is not null),
                ("SLOT_CTX", seq is null ? 0 : SlotContext(cacheWrite!, y)),
                .. cacheWrite is null ? [] : PageMacros(cacheWrite, pages),
            ];
            var rowKernel = Lib("rope.slang", "rope_row", Macros(fusedMacros));
            var weightOrX = normWeight ?? x;
            var factorsOrX = frequencyFactors ?? x;
            var nodes = new[] { norm, n, cacheWrite }.OfType<Node>();
            Emit($"rope{(norm is null ? "" : "+norm")}{(cacheWrite is null ? "" : "+cache_write")} {n.Output.Name}", [.. nodes],
                [Always(rowKernel, b => new Grid((uint)(Rows(b, tx) * H)))],
                [Pointer(x), Pointer(pos), Pointer(weightOrX), Pointer(y), new PointerArg(_runtime, 0), Pointer(factorsOrX), .. SeqArgs(seq, pages), TokenArg(x)],
                [.. (normWeight is null ? [x, pos] : (TensorRef[])[x, pos, normWeight]), .. frequencyFactors is null ? [] : (TensorRef[])[frequencyFactors],
                 .. PresentTables(seq, pages)], [y]);
            return;
        }

        long D = x.Shape[2];
        var kernel = Lib("rope.slang", "main", Macros(ropeMacros));
        Emit($"rope {n.Output.Name}", [n],
            [Always(kernel, b => new Grid(Grid.Div(Rows(b, tx) * H * (D / 2), 64)))],
            [Pointer(x), Pointer(pos), Pointer(y), Pointer(frequencyFactors ?? x), TokenArg(x)],
            frequencyFactors is null ? [x, pos] : [x, pos, frequencyFactors], [y]);
    }

    /// <summary>
    /// (token, head) rows per <c>rope_qkv</c> workgroup (<c>QKV_RPW</c>) below <see cref="RopeQkvLongRows"/> rows: one row per
    /// workgroup left the launch bound by how fast its 64-thread workgroups start. RTX PRO 4500, gpt-oss-20b (64 + 2 x 8 heads of 64
    /// values, no q / k norm), µs per launch for 1 / 4 / 8 / 16 rows per workgroup: 64 tokens 8.1 / 4.7 / 6.0 / 8.1, 2048 tokens
    /// 161.0 / 77.4 / 71.5 / 71.7 (benchmarks.md §80).
    /// </summary>
    private const int RopeQkvShortRowsPerGroup = 4;

    /// <summary>Rows per <c>rope_qkv</c> workgroup from <see cref="RopeQkvLongRows"/> rows on (see <see cref="RopeQkvShortRowsPerGroup"/>).</summary>
    private const int RopeQkvLongRowsPerGroup = 8;

    /// <summary>(token, head) rows of a <c>rope_qkv</c> launch from which it runs <see cref="RopeQkvLongRowsPerGroup"/> rows per workgroup.</summary>
    private const long RopeQkvLongRows = 32768;

    /// <summary>
    /// The query rope <paramref name="queryRope"/>, the key cache write <paramref name="keyWrite"/> with its rope and the value cache write
    /// <paramref name="valueWrite"/> of one attention layer as one <c>rope_qkv</c> launch (each with its absorbed RmsNorm), where the rows
    /// are [T, heads, D] with contiguous heads, both caches are f16 or f32 of one shape and row rule, and the two ropes differ only in
    /// their heads. Returns false, having emitted nothing, otherwise. Every value is computed as by the separate kernels (rope_row, rope,
    /// cache_write).
    /// </summary>
    private bool LowerRopeQkv(Node queryRope, Node keyWrite, Node valueWrite)
    {
        var keyRope = _absorbedRope[keyWrite.Inputs[1]];
        var queryNorm = _absorbedNorms.GetValueOrDefault(queryRope.Inputs[0]);
        var keyNorm = _absorbedNorms.GetValueOrDefault(keyRope.Inputs[0]);
        var xq = Ref(queryNorm?.Inputs[0] ?? queryRope.Inputs[0]);
        var xk = Ref(keyNorm?.Inputs[0] ?? keyRope.Inputs[0]);
        var xv = Ref(valueWrite.Inputs[1]);
        var kc = Ref(keyWrite.Inputs[0]);
        var vc = Ref(valueWrite.Inputs[0]);
        var pos = Ref(queryRope.Inputs[1]);
        TensorRef? NormWeight(Node? norm) => norm is not null && norm.Inputs.Length > 1 ? Ref(norm.Inputs[1]) : null;
        var wq = NormWeight(queryNorm);
        var wk = NormWeight(keyNorm);
        bool Rows3(TensorRef r) => r.Rank == 3 && r.Strides[2] == 1;
        if (!Rows3(xq) || !Rows3(xk) || !Rows3(xv) || kc.Rank != 3 || vc.Rank != 3 || kc.Type != vc.Type
            || (kc.Type != DataType.F16 && kc.Type != DataType.F32) || !pos.IsDense
            || (queryNorm is not null && wq is null) || (keyNorm is not null && wk is null)
            || xq.Symbol != xk.Symbol || xq.Symbol != xv.Symbol || xq.Shape[0] != xk.Shape[0] || xq.Shape[0] != xv.Shape[0])
        {
            return RopeQkvSkipped(queryRope, "row layouts or cache types");
        }

        long hq = xq.Shape[1];
        long hk = xk.Shape[1];
        long d = xq.Shape[2];
        if (xk.Shape[2] != d || xv.Shape[2] != d || xv.Shape[1] != hk || kc.Shape[1] != hk || kc.Shape[2] != d
            || vc.Shape[1] != hk || vc.Shape[2] != d || kc.Shape[0] != vc.Shape[0])
        {
            return RopeQkvSkipped(queryRope, "head shapes");
        }

        var frequencyFactors = queryRope.Inputs.Length > 2 ? Ref(queryRope.Inputs[2]) : null;
        var queryMacros = RopeMacros(queryRope.Attributes, xq, pos, frequencyFactors is not null);
        var keyMacros = RopeMacros(keyRope.Attributes, xk, pos, frequencyFactors is not null);
        string[] perHead = ["H", "S0", "S1"];
        if (!queryMacros.Where(m => !perHead.Contains(m.Item1)).SequenceEqual(keyMacros.Where(m => !perHead.Contains(m.Item1))))
        {
            return RopeQkvSkipped(queryRope, "rope attributes");
        }

        bool multiSeq = keyWrite.Inputs.Length > 2;
        if (!multiSeq && Slot(keyWrite) != Slot(valueWrite))
        {
            return RopeQkvSkipped(queryRope, "cache offsets");
        }

        if (keyWrite.Attributes.Get("block_size", 0L) != valueWrite.Attributes.Get("block_size", 0L)
            || (multiSeq && SlotContext(keyWrite, kc) != SlotContext(valueWrite, vc)))
        {
            return RopeQkvSkipped(queryRope, "cache layouts");
        }

        var seq = multiSeq ? DenseInput(keyWrite.Inputs[2], "seq-rows") : null;
        var pages = PageTable(keyWrite, 3);

        (string, object)[] macros =
        [
            .. queryMacros.Where(m => !perHead.Contains(m.Item1)),
            ("HQ", hq), ("HK", hk), ("QS0", xq.Strides[0]), ("QS1", xq.Strides[1]), ("KS0", xk.Strides[0]), ("KS1", xk.Strides[1]),
            ("VS0", xv.Strides[0]), ("VS1", xv.Strides[1]),
            ("NORM_Q", queryNorm is not null), ("EPS_Q", queryNorm?.Attributes.Get<float>("eps") ?? 0f),
            ("NORM_K", keyNorm is not null), ("EPS_K", keyNorm?.Attributes.Get<float>("eps") ?? 0f),
            ("CACHE_F16", kc.Type == DataType.F16),
            ("RT_N_PAST", seq is not null ? 0 : Slot(keyWrite)),
            ("MULTI_SEQ", seq is not null),
            ("SLOT_CTX", seq is null ? 0 : SlotContext(keyWrite, kc)),
            .. PageMacros(keyWrite, pages),
        ];
        var shortKernel = Lib("rope.slang", "rope_qkv", Macros([.. macros, ("QKV_RPW", RopeQkvShortRowsPerGroup)]));
        var longKernel = Lib("rope.slang", "rope_qkv", Macros([.. macros, ("QKV_RPW", RopeQkvLongRowsPerGroup)]));
        var yq = Produce(queryRope.Output);
        var tx = xq;
        long heads = hq + 2 * hk;
        Node[] nodes = [.. new[] { queryNorm, queryRope, keyNorm, keyRope }.OfType<Node>(), keyWrite, valueWrite];
        Emit($"rope+qkv+cache_write {queryRope.Output.Name}", nodes,
            [new KernelVariant(b => Rows(b, tx) * heads >= RopeQkvLongRows, longKernel, b => new Grid(Grid.Div(Rows(b, tx) * heads, RopeQkvLongRowsPerGroup))),
             Always(shortKernel, b => new Grid(Grid.Div(Rows(b, tx) * heads, RopeQkvShortRowsPerGroup)))],
            [Pointer(xq), Pointer(xk), Pointer(xv), Pointer(pos), Pointer(wq ?? xq), Pointer(wk ?? xk), Pointer(yq), Pointer(kc), Pointer(vc),
             new PointerArg(_runtime, 0), Pointer(frequencyFactors ?? xq), .. SeqArgs(seq, pages), TokenArg(xq)],
            [xq, xk, xv, pos, .. new[] { wq, wk, frequencyFactors }.OfType<TensorRef>(), .. PresentTables(seq, pages)], [yq, kc, vc]);
        return true;
    }

    /// <summary>False, after naming why under <c>SHARPLIGHT_ROPE_QKV_DEBUG=1</c>: the layer's ropes and cache writes stay separate launches.</summary>
    private static bool RopeQkvSkipped(Node queryRope, string why)
    {
        if (DiagnosticSwitches.RopeQkvDebug)
        {
            Console.Error.WriteLine($"rope q+k+v of {queryRope.Output.Name} not fused: {why}");
        }

        return false;
    }

    /// <summary>Kernel macros shared by the plain and the fused (norm / cache write) rope kernels.</summary>
    private static (string, object)[] RopeMacros(Attributes a, TensorRef x, TensorRef pos, bool hasFrequencyFactors)
    {
        long H = x.Shape[1];
        long D = x.Shape[2];
        int rotaryDims = a.Get("dims", (int)D);
        float theta = a.Get<float>("base");
        var sections = a.Get<int[]>("sections", []);
        bool mrope = sections.Length > 0 && sections.Any(s => s > 0);
        var sec = new int[4];
        for (int i = 0; i < Math.Min(4, sections.Length); i++)
        {
            sec[i] = sections[i];
        }

        long axes = pos.Rank > 1 ? pos.Shape[1] : 1;
        return
        [
            ("HAS_FF", hasFrequencyFactors),
            ("H", H), ("D", D), ("S0", x.Strides[0]), ("S1", x.Strides[1]), ("A", axes), ("ROT", rotaryDims),
            ("THETA_SCALE", (float)Math.Pow(theta, -2.0 / rotaryDims)), ("FREQ_SCALE", a.Get("freq_scale", 1f)), ("ATTN_FACTOR", a.Get("attn_factor", 1f)),
            ("NEOX", a.Get("style", "neox") == "neox"), ("MROPE", mrope), ("INTERLEAVED", a.Get("interleaved", false)),
            ("PER_ELEMENT", mrope && a.Get("sections_per_element", false)),
            ("SEC0", sec[0]), ("SEC1", sec[1]), ("SEC2", sec[2]), ("SEC3", sec[3]),
        ];
    }

    // ---- gather ------------------------------------------------------------------------------
    private void LowerGather(Node n)
    {
        var table = Ref(n.Inputs[0]);
        var idx = Ref(n.Inputs[1]);
        idx = EnsureDense(idx, "gather-idx");

        long D = table.Shape[^1];
        if (D % 8 != 0 && !table.Type.IsQuantized)
        {
            // Rows that are not a multiple of the kernel's 8-value groups (e.g. 180 channels): one thread per output value.
            string element = $"p.in0[uint64_t(clamp(p.in1[d0], 0, {table.Shape[0] - 1})) * {table.Strides[0]}u + d1]";
            string load = table.Type == DataType.F16 ? $"h2f({element})" : table.Type == DataType.BF16 ? $"bf2f({element})" : element;
            var rowsOut = Produce(n.Output);
            EmitGen($"gather {n.Output.Name}", [n], rowsOut.Shape, rowsOut.Symbol, [table, idx], [rowsOut], $"  p.out0[idx] = {load};");
            return;
        }

        long rowBytes;
        if (table.Type.IsQuantized || table.IsDense)
        {
            rowBytes = table.Type.ByteSize(D);
        }
        else
        {
            if (!TryRows(table, out long rowStride, out _) || (rowStride * 4) % 16 != 0 || table.ByteOffset % 16 != 0)
            {
                table = DenseTemp(table, "gather");
                rowStride = D;
            }

            rowBytes = rowStride * table.Type.ElementSize;
        }

        if (n.Output.Type != DataType.F32)
        {
            throw new NotSupportedException($"Gather to {n.Output.Type} (only f32 output is implemented).");
        }

        var y = Produce(n.Output);
        int fmt = WeightFormat(table.Type) ?? throw new NotSupportedException($"Gather table type {table.Type}");
        var k = Lib("data.slang", "gather", Macros(("D", D), ("WFMT", fmt), ("ROW_BYTES", rowBytes + "ull")));
        var ti = idx;
        Emit($"gather {n.Output.Name}", [n],
            [Always(k, b => new Grid(Grid.Div(Rows(b, ti) * (D / 8), 128)))],
            [Pointer(table), Pointer(idx), Pointer(y), TokenArg(idx)],
            [table, idx], [y]);
    }

    /// <summary>SequencePool: one thread per (sequence, column) loops over the sequence's rows.</summary>
    private void LowerSequencePool(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "seqpool");
        if (x.Rank != 2 || x.Strides[1] != 1)
        {
            x = DenseTemp(x, "seqpool");
        }

        var sp = DenseInput(n.Inputs[1], "seqpool-spans");
        var y = Produce(n.Output);
        string mode = n.Attributes.Get("mode", "mean");
        string load = $"p.in0[uint64_t(first + r) * {x.Strides[0]}u + d1]";
        string body = mode switch
        {
            "first" => $"  int first = p.in1[d0 * 2], cnt = p.in1[d0 * 2 + 1];\n  p.out0[idx] = cnt > 0 ? p.in0[uint64_t(first) * {x.Strides[0]}u + d1] : 0.0;",
            "last" => $"  int first = p.in1[d0 * 2], cnt = p.in1[d0 * 2 + 1];\n  p.out0[idx] = cnt > 0 ? p.in0[uint64_t(first + cnt - 1) * {x.Strides[0]}u + d1] : 0.0;",
            "max" => $"  int first = p.in1[d0 * 2], cnt = p.in1[d0 * 2 + 1];\n  float acc = -3.4028235e38;\n  for (int r = 0; r < cnt; r++) acc = max(acc, {load});\n  p.out0[idx] = cnt > 0 ? acc : 0.0;",
            _ => $"  int first = p.in1[d0 * 2], cnt = p.in1[d0 * 2 + 1];\n  float acc = 0.0;\n  for (int r = 0; r < cnt; r++) acc += {load};\n" +
                 (mode == "mean" ? "  p.out0[idx] = cnt > 0 ? acc / float(cnt) : 0.0;" : "  p.out0[idx] = acc;"),
        };
        EmitGen($"seqpool {mode} {n.Output.Name}", [n], y.Shape, y.Symbol, [x, sp], [y], body);
    }
}
