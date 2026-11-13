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
            EmitRowNormKernel($"{n.Op.Name} {n.Output.Name}", [n], x.Shape, x.Symbol, [x], [], "a0", [], [], [(y, "nrm")], w, mode, eps);
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
            ];
            var rowKernel = Lib("rope.slang", "rope_row", Macros(fusedMacros));
            var weightOrX = normWeight ?? x;
            var factorsOrX = frequencyFactors ?? x;
            var nodes = new[] { norm, n, cacheWrite }.OfType<Node>();
            Emit($"rope{(norm is null ? "" : "+norm")}{(cacheWrite is null ? "" : "+cache_write")} {n.Output.Name}", [.. nodes],
                [Always(rowKernel, b => new Grid((uint)(Rows(b, tx) * H)))],
                [Pointer(x), Pointer(pos), Pointer(weightOrX), Pointer(y), new PointerArg(_runtime, 0), Pointer(factorsOrX), .. SeqArgs(seq), TokenArg(x)],
                [.. (normWeight is null ? [x, pos] : (TensorRef[])[x, pos, normWeight]), .. frequencyFactors is null ? [] : (TensorRef[])[frequencyFactors],
                 .. seq is null ? [] : (TensorRef[])[seq]], [y]);
            return;
        }

        long D = x.Shape[2];
        var kernel = Lib("rope.slang", "main", Macros(ropeMacros));
        Emit($"rope {n.Output.Name}", [n],
            [Always(kernel, b => new Grid(Grid.Div(Rows(b, tx) * H * (D / 2), 64)))],
            [Pointer(x), Pointer(pos), Pointer(y), Pointer(frequencyFactors ?? x), TokenArg(x)],
            frequencyFactors is null ? [x, pos] : [x, pos, frequencyFactors], [y]);
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
            ("THETA_SCALE", (float)Math.Pow(theta, -2.0 / rotaryDims)), ("FREQ_SCALE", a.Get("freq_scale", 1f)),
            ("NEOX", a.Get("style", "neox") == "neox"), ("MROPE", mrope), ("INTERLEAVED", a.Get("interleaved", false)),
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
