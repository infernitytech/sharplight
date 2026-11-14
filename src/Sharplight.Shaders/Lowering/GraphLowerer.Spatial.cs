using System.Text;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Lowerings for general-network ops: Conv, Pool, Reduce, BatchMatMul, Pad, Resize and row ops
/// (Softmax, LayerNorm, ReduceSum/Max, ArgMax).</summary>
public sealed partial class GraphLowerer
{
    // A property (not a field) so it can be used from field initializers in other parts regardless of static init order.
    private static string[] SpatialOpNames => ["Conv", "Pool", "Reduce", "BatchMatMul", "Pad", "Resize", "Softmax", "LayerNorm", "ReduceSum", "ReduceMax", "ArgMax"];
    private static readonly HashSet<string> s_spatialOps = [.. SpatialOpNames];

    /// <summary>Support check for the ops handled in this file.</summary>
    private static OpSupport SupportsSpatial(Node node)
    {
        var a = node.Attributes;
        switch (node.Op.Name)
        {
            case "Conv":
                if (node.Inputs[0].Type.IsQuantized)
                {
                    return OpSupport.No("Conv input must not be quantized");
                }

                if (node.Inputs[1].Type.IsQuantized && WeightFormat(node.Inputs[1].Type) is null)
                {
                    return OpSupport.No($"weight format {node.Inputs[1].Type} not supported");
                }

                if (UsesGrid(node.Inputs[1].Type))
                {
                    return OpSupport.No($"Conv weight format {node.Inputs[1].Type} (lookup-grid formats are not supported by the convolution kernels)");
                }

                if (node.Inputs[1].Type != DataType.F32 && node.Inputs[1].Type != DataType.F16 && node.Inputs[1].Type != DataType.BF16 && !node.Inputs[1].Type.IsQuantized)
                {
                    return OpSupport.No($"Conv weight type {node.Inputs[1].Type}");
                }

                if (node.Inputs.Length > 2 && node.Inputs[2].Type.IsQuantized)
                {
                    return OpSupport.No("quantized bias");
                }

                return OpSupport.Yes;
            case "Resize":
                if (!(a.Get("mode", "nearest") is "nearest" or "linear"))
                {
                    return OpSupport.No($"Resize mode {a.Get("mode", "nearest")}");
                }

                if (!(a.Get("coord", "half_pixel") is "half_pixel" or "align_corners" or "asymmetric" or "pytorch_half_pixel"))
                {
                    return OpSupport.No($"Resize coord {a.Get("coord", "half_pixel")}");
                }

                if (!(a.Get("nearest_mode", "round_prefer_floor") is "round_prefer_floor" or "round_prefer_ceil" or "floor" or "ceil"))
                {
                    return OpSupport.No($"Resize nearest_mode {a.Get("nearest_mode", "round_prefer_floor")}");
                }

                break;
            case "Pad":
                if (!(a.Get("mode", "constant") is "constant" or "reflect" or "edge"))
                {
                    return OpSupport.No($"Pad mode {a.Get("mode", "constant")}");
                }

                break;
            case "Pool":
                if (!(a.Get("mode", "max") is "max" or "avg"))
                {
                    return OpSupport.No($"Pool mode {a.Get("mode", "max")}");
                }

                break;
            case "Reduce":
                if (!(a.Get("mode", "sum") is "sum" or "mean" or "max" or "min" or "prod" or "l1" or "l2" or "sumsquare" or "logsumexp"))
                {
                    return OpSupport.No($"Reduce mode {a.Get("mode", "sum")}");
                }

                break;
        }
        foreach (var v in node.Inputs)
        {
            if (v.Type.IsQuantized)
            {
                return OpSupport.No($"{node.Op.Name} cannot read quantized {v.Type}");
            }
            else if (v.Type != DataType.F32 && v.Type != DataType.F16 && v.Type != DataType.BF16)
            {
                return OpSupport.No($"{node.Op.Name} input type {v.Type}");
            }
        }

        return OpSupport.Yes;
    }

    private bool LowerSpatial(Node n)
    {
        switch (n.Op.Name)
        {
            case "Conv": LowerConv2d(n); return true;
            case "Pool": LowerPool(n); return true;
            case "Reduce": LowerReduce(n); return true;
            case "BatchMatMul": LowerBatchMatMul(n); return true;
            case "Pad": LowerPad(n); return true;
            case "Resize": LowerResize(n); return true;
            case "Softmax": LowerSoftmax(n); return true;
            case "LayerNorm": LowerLayerNorm(n); return true;
            case "ReduceSum": LowerRowReduce(n, 0); return true;
            case "ReduceMax": LowerRowReduce(n, 1); return true;
            case "ArgMax": LowerRowReduce(n, 2); return true;
        }
        return false;
    }

    // ---- helpers -----------------------------------------------------------------------------
    private static string IntList(IEnumerable<long> v)
    {
        var s = string.Join(",", v);
        return s.Length == 0 ? "0" : s;
    }

    /// <summary>Total tokens of the (possibly symbolic) outermost dim times <paramref name="inner"/> as a launch argument.</summary>
    private static KernelArg CountArg(TensorRef r, long inner) => r.Symbol is null ? new UIntArg((uint)(r.Shape[0] * inner)) : new SymbolArg(r.Symbol, inner);

    /// <summary>One variant per distinct choice over all bindings of the symbolic dim (1..bound).</summary>
    private List<KernelVariant> ChooseVariants<TCfg>(string? symbol, long bound, Func<long, TCfg> choose, Func<TCfg, (CompiledKernel Kernel, Func<long, Grid> Grid)> make)
        where TCfg : notnull
    {
        var list = new List<KernelVariant>();
        if (symbol is null)
        {
            var (k, g) = make(choose(bound));
            list.Add(Always(k, _ => g(bound)));
            return list;
        }
        var seen = new Dictionary<TCfg, bool>();
        for (long v = 1; v <= Math.Max(1, bound); v++)
        {
            var cfg = choose(v);
            if (!seen.TryAdd(cfg, true))
            {
                continue;
            }

            var (k, g) = make(cfg);
            var c = cfg;
            list.Add(new KernelVariant(b => choose(b[symbol]).Equals(c), k, b => g(b[symbol])));
        }
        return list;
    }

    // ---- Conv --------------------------------------------------------------------------------
    private sealed record ConvTile(int BM, int BN, int TM, int TN)
    {
        public int NT => (BM / TM) * (BN / TN);
    }

    private static readonly ConvTile[] s_convTiles =
        [new(128, 128, 8, 8), new(128, 64, 8, 4), new(64, 64, 4, 4), new(64, 32, 4, 4), new(32, 64, 4, 4), new(32, 32, 4, 4)];
    private const int ConvBK = 32;
    private const long TargetWorkgroups = 480;

    private sealed record ConvCfg(ConvTile Tile, int Split, long Kps);

    private static ConvCfg ChooseConv(long N, long OG, long G, long HW, long K)
    {
        long NC = N * HW;
        ConvTile? best = null;
        long bestWgs = 0;
        foreach (var t in s_convTiles)
        {
            // Avoid grossly oversized tiles for small problems.
            if (t.BM > 2 * Math.Max(OG, 16) && t.BM > 32)
            {
                continue;
            }

            if (t.BN > 2 * Math.Max(NC, 16) && t.BN > 32)
            {
                continue;
            }

            long wgs = ((NC + t.BN - 1) / t.BN) * ((OG + t.BM - 1) / t.BM) * G;
            if (wgs >= TargetWorkgroups)
            {
                return new ConvCfg(t, 1, K);
            }

            if (wgs > bestWgs)
            {
                best = t;
                bestWgs = wgs;
            }
        }
        best ??= s_convTiles[^1];
        bestWgs = Math.Max(1, bestWgs);
        long kTiles = (K + ConvBK - 1) / ConvBK;
        long split = Math.Min(Math.Min((TargetWorkgroups + bestWgs - 1) / bestWgs, kTiles / 4), 16);
        if (split < 2)
        {
            return new ConvCfg(best, 1, K);
        }

        long kps = (kTiles + split - 1) / split * ConvBK;
        split = (K + kps - 1) / kps;
        return new ConvCfg(best, (int)split, kps);
    }

    /// <summary>
    /// The kernel <paramref name="entry"/> of a convolution file, with <paramref name="epilogue"/>'s group applied in its
    /// final store when given (<c>conv_epilogue.slang</c>).
    /// </summary>
    private CompiledKernel ConvKernel(string file, string entry, Dictionary<string, string> macros, ConvEpilogueCode? epilogue)
    {
        if (epilogue is null)
        {
            return Lib(file, entry, macros);
        }

        var m = new Dictionary<string, string>(macros)
        {
            ["KERNEL_" + entry.ToUpperInvariant()] = "1",
            ["EPI"] = "1",
            ["EPI_RES"] = epilogue.Residual is null ? "0" : "1",
        };
        return _compiler.Compile(file.Replace(".slang", "_epi.slang", StringComparison.Ordinal), epilogue.Source + $"#include \"{file}\"\n", entry, _opt.Target, m, _opt.Cache);
    }

    /// <summary>Generated <c>conv_epi</c> source of a fused group and its residual input (dense f32 of y's shape), if any.</summary>
    private sealed record ConvEpilogueCode(string Source, TensorRef? Residual);

    /// <summary>The <c>conv_epi(m0, r0)</c> function computing <paramref name="epilogue"/>'s group from the conv value m0.</summary>
    private ConvEpilogueCode EpilogueCode(Node conv, ConvEpilogue epilogue)
    {
        var name = new Dictionary<Value, string> { [conv.Output] = "m0" };
        var inGroup = epilogue.Group.Select(n => n.Output).ToHashSet();
        var residual = epilogue.Group.SelectMany(n => n.Inputs).FirstOrDefault(v => v != conv.Output && !inGroup.Contains(v));
        if (residual is not null)
        {
            name[residual] = "r0";
        }

        var src = new StringBuilder();
        src.AppendLine("#include \"math_ext.slang\"");
        src.AppendLine("float conv_epi(float m0, float r0) {");
        foreach (var node in epilogue.Group)
        {
            var local = "v" + name.Count;
            src.AppendLine($"  float {local} = {OpExpr(node, [.. node.Inputs.Select(i => name[i])])};");
            name[node.Output] = local;
        }

        src.AppendLine($"  return {name[epilogue.Output]};");
        src.AppendLine("}");
        return new ConvEpilogueCode(src.ToString(), residual is null ? null : AsDenseF32(Ref(residual), "conv-res"));
    }

    /// <summary>
    /// Lowers a 2-D convolution: depthwise, cooperative-matrix implicit GEMM, or the SIMT implicit GEMM (split-K through a
    /// workspace and a reduce). With <paramref name="fused"/>, the final store applies that elementwise group and writes its output.
    /// </summary>
    private void LowerConv2d(Node n, ConvEpilogue? fused = null)
    {
        var epilogue = fused is null ? null : EpilogueCode(n, fused);
        Node[] nodes = fused is null ? [n] : [n, .. fused.Group];
        string fusedLabel = fused is null ? "" : $" fused[{string.Join(",", fused.Group.Select(g => g.Op.Name))}]";
        var resArgs = epilogue?.Residual is { } resRef ? new KernelArg[] { Pointer(resRef) } : [];
        var resReads = epilogue?.Residual is { } resRead ? new[] { resRead } : [];
        var x = AsF32(Ref(n.Inputs[0]), "conv-x");
        var w = Ref(n.Inputs[1]);
        if (!w.IsDense)
        {
            if (w.Type.IsQuantized)
            {
                throw new NotSupportedException("Conv weight must be dense.");
            }

            w = DenseTemp(w, "conv-w");
        }
        TensorRef? bias = n.Inputs.Length > 2 ? AsDenseF32(Ref(n.Inputs[2]), "conv-bias") : null;
        var a = n.Attributes;
        var st = a.Get("strides", new long[] { 1, 1 });
        var pd = a.Get("pads", new long[] { 0, 0, 0, 0 });
        var dl = a.Get("dilations", new long[] { 1, 1 });
        long G = a.Get("groups", 1L);
        long C = x.Shape[1];
        long H = x.Shape[2];
        long W = x.Shape[3];
        long O = w.Shape[0];
        long CG = w.Shape[1];
        long KH = w.Shape[2];
        long KW = w.Shape[3];
        long OG = O / G;
        var y = Produce(fused?.Output ?? n.Output);
        long HO = y.Shape[2];
        long WO = y.Shape[3];
        long HW = HO * WO;
        long K = CG * KH * KW;
        int fmt = WeightFormat(w.Type) ?? throw new NotSupportedException($"Conv weight type {w.Type}");
        bool nopad = pd.All(p => p == 0);
        var common = new (string, object)[]
        {
            ("C", C), ("H", H), ("W", W), ("O", O), ("HO", HO), ("WO", WO), ("CG", CG), ("OG", OG), ("G", G), ("KH", KH), ("KW", KW),
            ("SH", st[0]), ("SW", st[1]), ("PH", pd[0]), ("PW", pd[1]), ("DH", dl[0]), ("DW", dl[1]),
            ("XS_N", x.Strides[0]), ("XS_C", x.Strides[1]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]),
            ("WFMT", fmt), ("ROW_BYTES", w.Type.ByteSize(K) + "ull"), ("HAS_BIAS", bias is not null), ("NOPAD", nopad),
        };
        var xr = x;
        var biasPtr = bias ?? x;
        string name = $"conv {n.Output.Name} [{C}→{O} {KH}x{KW} s{st[0]} g{G}] {w.Type}{fusedLabel}";

        // Depthwise (channel multiplier OG) on float weights.
        if (G == C && CG == 1 && !w.Type.IsQuantized)
        {
            var k = ConvKernel("conv.slang", "conv_dw", Macros(common), epilogue);
            Emit(name, nodes, [Always(k, b => new Grid(Grid.Div(Rows(b, xr) * O * HW, 256)))],
                [Pointer(x), Pointer(w), Pointer(biasPtr), Pointer(y), .. resArgs, CountArg(x, O * HW)],
                [.. bias is null ? [x, w] : new[] { x, w, bias }, .. resReads], [y]);
            return;
        }

        if (TryLowerConvMatrix(n, nodes, epilogue, x, w, bias, y, common, name))
        {
            return;
        }

        bool aVec = w.Type.IsQuantized || K % 8 == 0;
        long bound = x.Symbol is null ? x.Shape[0] : Bound(Dim.Sym(x.Symbol));
        Func<long, ConvCfg> choose = N => ChooseConv(N, OG, G, HW, K);
        long maxWs = 0;
        for (long v = x.Symbol is null ? bound : 1; v <= bound; v++)
        {
            var c = choose(v);
            if (c.Split > 1)
            {
                maxWs = Math.Max(maxWs, c.Split * v * O * HW);
            }
        }
        TensorRef? ws = maxWs > 0 ? Temp(DataType.F32, [maxWs], null) : null;

        bool useWs = maxWs > 0;
        // With split-K present for some bindings every variant goes through the workspace and the reduce
        // kernel (which adds the bias); split-1 variants run both kernels with SPLITK=1.
        var gemmVariants = ChooseVariants(x.Symbol, bound, choose, cfg =>
        {
            var t = cfg.Tile;
            var k = ConvKernel("conv.slang", "conv_gemm", Macros([.. common.Where(c => c.Item1 != "HAS_BIAS"), ("HAS_BIAS", bias is not null && !useWs),
                ("BM", t.BM), ("BN", t.BN), ("BK", ConvBK), ("TM", t.TM), ("TN", t.TN), ("NT", t.NT), ("A_VEC", aVec), ("SPLITK", cfg.Split), ("KPS", cfg.Kps)]),
                useWs ? null : epilogue);
            return (k, Nv => new Grid(Grid.Div(Nv * HW, t.BN), Grid.Div(OG, t.BM), (uint)(G * cfg.Split)));
        });
        var dst = ws ?? y;
        Emit(useWs ? name + " gemm" : name, nodes, gemmVariants,
            [Pointer(x), Pointer(w), Pointer(biasPtr), Pointer(dst), .. useWs ? [] : resArgs, TokenArg(x)],
            [.. bias is null || useWs ? [x, w] : new[] { x, w, bias }, .. useWs ? [] : resReads], [dst]);
        if (!useWs)
        {
            return;
        }

        var redVariants = ChooseVariants(x.Symbol, bound, choose, cfg =>
        {
            var k = ConvKernel("conv.slang", "conv_splitk_reduce", Macros(("SPLITK", cfg.Split), ("O", O), ("HW", HW), ("HAS_BIAS", bias is not null)), epilogue);
            return (k, Nv => new Grid(Grid.Div(Nv * O * HW, 256)));
        });
        Emit(name + " reduce", nodes, redVariants,
            [Pointer(dst), Pointer(biasPtr), Pointer(y), .. resArgs, CountArg(x, O * HW)],
            [.. bias is null ? [dst] : new[] { dst, bias }, .. resReads], [y]);
    }

    // ---- Pool --------------------------------------------------------------------------------
    private void LowerPool(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "pool");
        var a = n.Attributes;
        var kernel = a.Get<long[]>("kernel");
        var st = a.Get("strides", new long[] { 1, 1 });
        var pd = a.Get("pads", new long[] { 0, 0, 0, 0 });
        var dl = a.Get("dilations", new long[] { 1, 1 });
        bool avg = a.Get("mode", "max") == "avg";
        var y = Produce(n.Output);
        long C = x.Shape[1];
        long HO = y.Shape[2];
        long WO = y.Shape[3];
        var k = Lib("spatial.slang", "pool", Macros(("C", C), ("H", x.Shape[2]), ("W", x.Shape[3]), ("HO", HO), ("WO", WO), ("KH", kernel[0]), ("KW", kernel[1]),
            ("SH", st[0]), ("SW", st[1]), ("PH", pd[0]), ("PW", pd[1]), ("PB", pd[2]), ("PR", pd[3]), ("DH", dl[0]), ("DW", dl[1]),
            ("XS_N", x.Strides[0]), ("XS_C", x.Strides[1]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]),
            ("MODE", avg ? 1 : 0), ("COUNT_PAD", a.Get("count_include_pad", false))));
        var tx = x;
        Emit($"pool {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * C * HO * WO, 256)))],
            [Pointer(x), Pointer(y), CountArg(x, C * HO * WO)], [x], [y]);
    }

    // ---- Reduce ------------------------------------------------------------------------------
    private void LowerReduce(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "reduce");
        var a = n.Attributes;
        var axes = a.Get<long[]>("axes").Select(ax => (int)(ax < 0 ? ax + x.Rank : ax)).ToHashSet();
        string mode = a.Get("mode", "sum");
        var y = Produce(n.Output);
        // Split dims into kept / reduced groups, merging adjacent dims of the same class with compatible strides.
        var kept = new List<(long Shape, long Stride)>();
        var red = new List<(long Shape, long Stride)>();
        for (int d = 0; d < x.Rank; d++)
        {
            bool isRed = axes.Contains(d);
            var list = isRed ? red : kept;
            if (x.Shape[d] == 1 && !(d == 0 && x.Symbol is not null))
            {
                continue;
            }
            // adjacent dims of one class merge when their addresses are contiguous
            if (list.Count > 0 && list[^1].Stride == x.Strides[d] * x.Shape[d])
            {
                list[^1] = (list[^1].Shape * x.Shape[d], x.Strides[d]);
            }
            else
            {
                list.Add((x.Shape[d], x.Strides[d]));
            }
        }
        long rTotal = red.Aggregate(1L, (acc, r) => acc * r.Shape);
        // The outermost kept (or reduced) dim is the quotient; with a symbolic dim its (bound) shape is never used in a modulo.
        bool sym = x.Symbol is not null;
        bool symReduced = sym && axes.Contains(0);
        bool symKept = sym && !symReduced;
        int tpo = rTotal >= 2048 ? 256 : _opt.SubgroupSize;
        var k = Lib("spatial.slang", "reduce", Macros(
            ("KN", Math.Max(1, kept.Count)), ("KSH", kept.Count == 0 ? "1" : IntList(kept.Select(t => t.Shape))), ("KST", kept.Count == 0 ? "0" : IntList(kept.Select(t => t.Stride))),
            ("RN", Math.Max(1, red.Count)), ("RSH", red.Count == 0 ? "1" : IntList(red.Select(t => t.Shape))), ("RST", red.Count == 0 ? "0" : IntList(red.Select(t => t.Stride))),
            ("R_TOTAL", symReduced ? "p.nRed" : rTotal), ("MODE", mode switch { "sum" => 0, "mean" => 1, "max" => 2, "min" => 3, "prod" => 4, "l1" => 5, "l2" => 6, "sumsquare" => 7, _ => 8 }), ("TPO", tpo)));
        long inner = y.Shape.Skip(1).Aggregate(1L, (acc, v) => acc * v);
        var tx = x;
        long perWg = tpo == 256 ? 1 : 256 / tpo;
        KernelArg count = symKept ? new SymbolArg(x.Symbol!, inner) : new UIntArg((uint)y.MaxElements);
        KernelArg reduced = symReduced ? new SymbolArg(x.Symbol!, rTotal / Math.Max(1, x.Shape[0])) : new UIntArg((uint)rTotal);
        Emit($"reduce-{mode} {n.Output.Name}", [n],
            [Always(k, b => new Grid(Grid.Div(symKept ? b[tx.Symbol!] * inner : y.MaxElements, perWg)))],
            [Pointer(x), Pointer(y), count, reduced], [x], [y]);
    }

    // ---- BatchMatMul -------------------------------------------------------------------------
    private sealed record BmmTile(int BM, int BN, int TM, int TN, int BK, int Split = 1, long Kps = 0)
    {
        public int NT => (BM / TM) * (BN / TN);
    }

    /// <summary>Smallest K worth splitting across workgroups in bmm (each split keeps at least <see cref="BmmMinKPerSplit"/>).</summary>
    private const long BmmSplitMinK = 512;

    /// <summary>K values per bmm split at least.</summary>
    private const long BmmMinKPerSplit = 256;

    private void LowerBatchMatMul(Node n)
    {
        var at = n.Attributes;
        LowerBmm(n, Ref(n.Inputs[0]), Ref(n.Inputs[1]), at.Get("trans_a", false), at.Get("trans_b", false), at.Get("alpha", 1f));
    }

    /// <summary>General strided (batched) matmul y = alpha · op(a) · op(b); also used for MatMul shapes the weight kernels can't take.</summary>
    private void LowerBmm(Node n, TensorRef aRef, TensorRef bRef, bool ta, bool tb, float alpha)
    {
        var a = AsF32(aRef, "bmm-a");
        var b = AsF32(bRef, "bmm-b");
        var y = Produce(n.Output);
        int nb = y.Rank - 2;
        long M = y.Shape[^2];
        long N = y.Shape[^1];
        long K = ta ? a.Shape[^2] : a.Shape[^1];
        long sam = ta ? a.Strides[^1] : a.Strides[^2];
        long sak = ta ? a.Strides[^2] : a.Strides[^1];
        long sbk = tb ? b.Strides[^1] : b.Strides[^2];
        long sbn = tb ? b.Strides[^2] : b.Strides[^1];
        // Batch dims, right-aligned broadcast against the output.
        var bsh = new long[nb];
        var bsa = new long[nb];
        var bsb = new long[nb];
        for (int d = 0; d < nb; d++)
        {
            bsh[d] = y.Shape[d];
            int ia = a.Rank - 2 - (nb - d);
            int ib = b.Rank - 2 - (nb - d);
            bsa[d] = ia >= 0 && a.Shape[ia] != 1 ? a.Strides[ia] : 0;
            bsb[d] = ib >= 0 && b.Shape[ib] != 1 ? b.Strides[ib] : 0;
        }
        bool mdyn = y.Symbol is not null && nb == 0;
        long batchesInner = 1;
        for (int d = 1; d < nb; d++)
        {
            batchesInner *= bsh[d];
        }

        long Mb = M;
        Func<long, BmmTile> choose = dyn =>
        {
            long batches = nb == 0 ? 1 : (y.Symbol is not null ? dyn * batchesInner : y.Shape.Take(nb).Aggregate(1L, (p, q) => p * q));
            long mm = mdyn ? dyn : Mb;
            long wgs64 = batches * ((mm + 63) / 64) * ((N + 63) / 64);
            long wgs32 = batches * ((mm + 31) / 32) * ((N + 31) / 32);
            // Small grids (attention P·V at batch 1: 12 heads x 7 x 2 tiles) are latency-bound with 64-thread groups;
            // 2x2 outputs per thread give each group 256 threads for the same tile.
            var tile = wgs64 >= 120 ? new BmmTile(64, 64, 4, 4, 16) : wgs32 >= 512 ? new BmmTile(32, 32, 4, 4, 16) : new BmmTile(32, 32, 2, 2, 16);
            // Few tiles over a long K (P·V of long sequences: whisper's 1500 keys, 6 heads x 24 tiles) leave most of the
            // device idle: split K towards TargetWorkgroups, the partials summed by bmm_reduce.
            long tiles = batches * ((mm + tile.BM - 1) / tile.BM) * ((N + tile.BN - 1) / tile.BN);
            long split = K < BmmSplitMinK || tiles >= TargetWorkgroups / 2 ? 1 : Math.Min(Math.Min((TargetWorkgroups + tiles - 1) / tiles, K / BmmMinKPerSplit), 8);
            if (split <= 1)
            {
                return tile;
            }

            long kSteps = (K + tile.BK - 1) / tile.BK;
            long kps = (kSteps + split - 1) / split * tile.BK;
            return tile with { Split = (int)((K + kps - 1) / kps), Kps = kps };
        };
        long bound = y.Symbol is null ? 1 : Bound(Dim.Sym(y.Symbol));
        int maxSplit = 1;
        for (long v = y.Symbol is null ? bound : 1; v <= bound; v++)
        {
            maxSplit = Math.Max(maxSplit, choose(v).Split);
        }

        bool useWs = maxSplit > 1;
        long wsStride = y.MaxElements;
        var dst = useWs ? Temp(DataType.F32, [maxSplit * wsStride], null) : y;
        var variants = ChooseVariants(y.Symbol, bound, choose, t =>
        {
            var k = Lib("bmm.slang", "bmm", Macros(("M", M), ("N", N), ("K", K), ("NB", nb), ("BSH", IntList(bsh)), ("BSA", IntList(bsa)), ("BSB", IntList(bsb)),
                ("SAM", sam), ("SAK", sak), ("SBK", sbk), ("SBN", sbn), ("ALPHA", alpha), ("M_DYN", mdyn),
                ("BM", t.BM), ("BN", t.BN), ("BK", t.BK), ("TM", t.TM), ("TN", t.TN), ("NT", t.NT),
                ("SPLITK", t.Split), ("KPS", t.Split > 1 ? t.Kps : K), ("WS_STRIDE", useWs ? wsStride + "ull" : "0"), ("PARTIAL", useWs)));
            return (k, dyn =>
            {
                long mm = mdyn ? dyn : Mb;
                long batches = nb == 0 ? 1 : (y.Symbol is not null ? dyn * batchesInner : y.Shape.Take(nb).Aggregate(1L, (p, q) => p * q));
                return new Grid(Grid.Div(N, t.BN), Grid.Div(mm, t.BM), (uint)(batches * t.Split));
            }
            );
        });
        KernelArg dynArg = mdyn ? new SymbolArg(y.Symbol!) : new UIntArg(0);
        string name = $"bmm {n.Output.Name} [{M}x{K}x{N}]";
        Emit(useWs ? name + " split-k" : name, [n], variants,
            [Pointer(a), Pointer(b), Pointer(dst), dynArg], [a, b], [dst]);
        if (!useWs)
        {
            return;
        }

        long inner = y.MaxElements / Math.Max(1, y.Shape[0]);
        var reduceVariants = ChooseVariants(y.Symbol, bound, choose, t =>
        {
            var k = Lib("bmm.slang", "bmm_reduce", Macros(("SPLITK", t.Split), ("KPS", K), ("WS_STRIDE", wsStride + "ull"), ("PARTIAL", 1), ("ALPHA", alpha)));
            return (k, dyn => new Grid(Grid.Div((y.Symbol is null ? y.Shape[0] : dyn) * inner, 256)));
        });
        Emit(name + " reduce", [n], reduceVariants, [Pointer(dst), Pointer(y), CountArg(y, inner)], [dst], [y]);
    }

    // ---- Pad ---------------------------------------------------------------------------------
    private void LowerPad(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "pad");
        var a = n.Attributes;
        var pads = a.Get<long[]>("pads");
        string mode = a.Get("mode", "constant");
        var y = Produce(n.Output);
        int r = x.Rank;
        var ish = x.Shape.Select((s, d) => d == 0 && x.Symbol is not null ? int.MaxValue : s).Select(v => (long)v);
        var k = Lib("spatial.slang", "pad", Macros(("RANK", Math.Max(1, r)), ("OSH", IntList(y.Shape)), ("ISH", IntList(ish)), ("IST", IntList(x.Strides)),
            ("PBG", IntList(pads.Take(r))), ("MODE", mode switch { "constant" => 0, "reflect" => 1, _ => 2 }), ("VALUE", a.Get("value", 0f))));
        long inner = y.Shape.Skip(1).Aggregate(1L, (p, q) => p * q);
        var tx = x;
        Emit($"pad {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * inner, 256)))],
            [Pointer(x), Pointer(y), CountArg(y, inner)], [x], [y]);
    }

    // ---- Resize ------------------------------------------------------------------------------
    private void LowerResize(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "resize");
        var a = n.Attributes;
        var y = Produce(n.Output);
        long C = x.Shape[1];
        long OH = y.Shape[2];
        long OW = y.Shape[3];
        int coord = a.Get("coord", "half_pixel") switch { "half_pixel" => 0, "align_corners" => 1, "asymmetric" => 2, _ => 3 };
        int near = a.Get("nearest_mode", "round_prefer_floor") switch { "round_prefer_floor" => 0, "round_prefer_ceil" => 1, "floor" => 2, _ => 3 };
        var k = Lib("spatial.slang", "resize", Macros(("C", C), ("H", x.Shape[2]), ("W", x.Shape[3]), ("OH", OH), ("OW", OW),
            ("XS_N", x.Strides[0]), ("XS_C", x.Strides[1]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]),
            ("LINEAR", a.Get("mode", "nearest") == "linear"), ("COORD", coord), ("NEAREST", near)));
        var tx = x;
        Emit($"resize {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * C * OH * OW, 256)))],
            [Pointer(x), Pointer(y), CountArg(x, C * OH * OW)], [x], [y]);
    }

    // ---- row ops -----------------------------------------------------------------------------
    private TensorRef RowsInput(Value v, string why, out long r1, out long s0, out long s1)
    {
        var x = AsF32(Ref(v), why);
        if (!TryRows3(x, out r1, out s0, out s1))
        {
            x = DenseTemp(x, why);
            TryRows3(x, out r1, out s0, out s1);
        }
        return x;
    }

    private void LowerSoftmax(Node n)
    {
        var x = RowsInput(n.Inputs[0], "softmax", out long r1, out long s0, out long s1);
        var y = Produce(n.Output);
        if (TryLowerSoftmaxSplit(n, x, y, r1, s0, s1))
        {
            return;
        }

        var k = Lib("rows.slang", "softmax", Macros(("D", x.Shape[^1]), ("R1", r1), ("S0", s0), ("S1", s1), ("SCALE", n.Attributes.Get("scale", 1f)), ("LOG_SOFTMAX", n.Attributes.Get("log", false))));
        var tx = x;
        Emit($"softmax {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * r1, 4)))],
            [Pointer(x), Pointer(y), TokenArg(x)], [x], [y]);
    }

    // LayerNorm rows served by one workgroup each (up to this many rows, of 512 to 16384 values).
    private const long LayerNormWorkgroupMaxRows = 64;
    private const long LayerNormWorkgroupMinWidth = 512;
    private const long LayerNormWorkgroupMaxWidth = 16384;

    private void LowerLayerNorm(Node n)
    {
        var x = RowsInput(n.Inputs[0], "layernorm", out long r1, out long s0, out long s1);
        var w = n.Inputs.Length > 1 ? AsDenseF32(Ref(n.Inputs[1]), "ln-w") : null;
        var bi = n.Inputs.Length > 2 ? AsDenseF32(Ref(n.Inputs[2]), "ln-b") : null;
        var y = Produce(n.Output);
        var macros = Macros(("D", x.Shape[^1]), ("R1", r1), ("S0", s0), ("S1", s1), ("EPS", n.Attributes.Get<float>("eps")),
            ("HAS_W", w is not null), ("HAS_B", bi is not null));
        var k = Lib("rows.slang", "layernorm", macros);
        var tx = x;
        var wp = w ?? x;
        var bp = bi ?? x;
        List<KernelVariant> variants = [];
        // Few wide rows (decode): a workgroup per row instead of a subgroup, which is latency bound on long rows.
        if (x.Shape[^1] >= LayerNormWorkgroupMinWidth && x.Shape[^1] <= LayerNormWorkgroupMaxWidth)
        {
            variants.Add(new KernelVariant(b => Rows(b, tx) * r1 <= LayerNormWorkgroupMaxRows, Lib("rows.slang", "layernorm_wg", macros),
                b => new Grid((uint)Math.Max(1, Rows(b, tx) * r1))));
        }

        variants.Add(Always(k, b => new Grid(Grid.Div(Rows(b, tx) * r1, 4))));
        Emit($"layernorm {n.Output.Name}", [n], variants,
            [Pointer(x), Pointer(wp), Pointer(bp), Pointer(y), TokenArg(x)],
            [.. new[] { x, w, bi }.Where(t => t is not null).Select(t => t!)], [y]);
    }

    /// <summary>Widest row reduced by one thread per row (rowreduce_narrow).</summary>
    private const long NarrowReduceMaxWidth = 32;

    private void LowerRowReduce(Node n, int mode)
    {
        var src = AsF32(Ref(n.Inputs[0]), "rowreduce");
        if (src.Rank is 2 or 3 && src.Shape[^1] <= NarrowReduceMaxWidth)
        {
            // Short rows: thread per row on the (possibly permuted) view itself.
            long rows1 = src.Rank == 3 ? src.Shape[1] : 1;
            var ny = Produce(n.Output);
            var nk = Lib("rows.slang", "rowreduce_narrow", Macros(("D", src.Shape[^1]), ("R1", rows1), ("S0", src.Strides[0]),
                ("S1", src.Rank == 3 ? src.Strides[1] : 0), ("SI", src.Strides[^1]), ("MODE", mode)));
            var ns = src;
            Emit($"{n.Op.Name} {n.Output.Name}", [n], [Always(nk, b => new Grid(Grid.Div(Rows(b, ns) * rows1, 256)))],
                [Pointer(src), Pointer(ny), TokenArg(src)], [src], [ny]);
            return;
        }

        var x = src;
        if (!TryRows3(x, out long r1, out long s0, out long s1))
        {
            x = DenseTemp(x, "rowreduce");
            TryRows3(x, out r1, out s0, out s1);
        }

        var y = Produce(n.Output);
        var k = Lib("rows.slang", "rowreduce", Macros(("D", x.Shape[^1]), ("R1", r1), ("S0", s0), ("S1", s1), ("MODE", mode)));
        var tx = x;
        Emit($"{n.Op.Name} {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * r1, 4)))],
            [Pointer(x), Pointer(y), TokenArg(x)], [x], [y]);
    }
}
