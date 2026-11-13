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

    private void LowerConv2d(Node n)
    {
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
        var y = Produce(n.Output);
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
        string name = $"conv {n.Output.Name} [{C}→{O} {KH}x{KW} s{st[0]} g{G}] {w.Type}";

        // Depthwise (channel multiplier OG) on float weights.
        if (G == C && CG == 1 && !w.Type.IsQuantized)
        {
            var k = Lib("conv.slang", "conv_dw", Macros(common));
            Emit(name, [n], [Always(k, b => new Grid(Grid.Div(Rows(b, xr) * O * HW, 256)))],
                [Pointer(x), Pointer(w), Pointer(biasPtr), Pointer(y),
                 CountArg(x, O * HW)],
                bias is null ? [x, w] : [x, w, bias], [y]);
            return;
        }

        if (TryLowerConvMatrix(n, x, w, bias, y, common, name))
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
            var k = Lib("conv.slang", "conv_gemm", Macros([.. common.Where(c => c.Item1 != "HAS_BIAS"), ("HAS_BIAS", bias is not null && !useWs),
                ("BM", t.BM), ("BN", t.BN), ("BK", ConvBK), ("TM", t.TM), ("TN", t.TN), ("NT", t.NT), ("A_VEC", aVec), ("SPLITK", cfg.Split), ("KPS", cfg.Kps)]));
            return (k, Nv => new Grid(Grid.Div(Nv * HW, t.BN), Grid.Div(OG, t.BM), (uint)(G * cfg.Split)));
        });
        var dst = ws ?? y;
        Emit(useWs ? name + " gemm" : name, [n], gemmVariants,
            [Pointer(x), Pointer(w), Pointer(biasPtr), Pointer(dst), TokenArg(x)],
            bias is null || useWs ? [x, w] : [x, w, bias], [dst]);
        if (!useWs)
        {
            return;
        }

        var redVariants = ChooseVariants(x.Symbol, bound, choose, cfg =>
        {
            var k = Lib("conv.slang", "conv_splitk_reduce", Macros(("SPLITK", cfg.Split), ("O", O), ("HW", HW), ("HAS_BIAS", bias is not null)));
            return (k, Nv => new Grid(Grid.Div(Nv * O * HW, 256)));
        });
        Emit(name + " reduce", [n], redVariants,
            [Pointer(dst), Pointer(biasPtr), Pointer(y), CountArg(x, O * HW)],
            bias is null ? [dst] : [dst, bias], [y]);
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
            ("SH", st[0]), ("SW", st[1]), ("PH", pd[0]), ("PW", pd[1]), ("PB", pd[2]), ("PR", pd[3]), ("DH", avg ? 1 : dl[0]), ("DW", avg ? 1 : dl[1]),
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
        long nOutInner = kept.Skip(x.Symbol is not null ? 1 : 0).Aggregate(1L, (acc, r) => acc * r.Shape);
        bool sym = x.Symbol is not null;
        // Outermost kept dim is the quotient; with a symbolic dim its (bound) shape is never used in a modulo.
        int tpo = rTotal >= 2048 ? 256 : _opt.SubgroupSize;
        var k = Lib("spatial.slang", "reduce", Macros(
            ("KN", Math.Max(1, kept.Count)), ("KSH", kept.Count == 0 ? "1" : IntList(kept.Select(t => t.Shape))), ("KST", kept.Count == 0 ? "0" : IntList(kept.Select(t => t.Stride))),
            ("RN", Math.Max(1, red.Count)), ("RSH", red.Count == 0 ? "1" : IntList(red.Select(t => t.Shape))), ("RST", red.Count == 0 ? "0" : IntList(red.Select(t => t.Stride))),
            ("R_TOTAL", rTotal), ("MODE", mode switch { "sum" => 0, "mean" => 1, "max" => 2, "min" => 3, "prod" => 4, "l1" => 5, "l2" => 6, "sumsquare" => 7, _ => 8 }), ("TPO", tpo)));
        long inner = y.Shape.Skip(1).Aggregate(1L, (acc, v) => acc * v);
        var tx = x;
        long perWg = tpo == 256 ? 1 : 256 / tpo;
        KernelArg count = sym ? new SymbolArg(x.Symbol!, inner) : new UIntArg((uint)(y.MaxElements));
        Emit($"reduce-{mode} {n.Output.Name}", [n],
            [Always(k, b => new Grid(Grid.Div((sym ? b[tx.Symbol!] * inner : y.MaxElements), perWg)))],
            [Pointer(x), Pointer(y), count], [x], [y]);
    }

    // ---- BatchMatMul -------------------------------------------------------------------------
    private sealed record BmmTile(int BM, int BN, int TM, int TN, int BK)
    {
        public int NT => (BM / TM) * (BN / TN);
    }

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
            return wgs64 >= 120 ? new BmmTile(64, 64, 4, 4, 16) : new BmmTile(32, 32, 4, 4, 16);
        };
        long bound = y.Symbol is null ? 1 : Bound(Dim.Sym(y.Symbol));
        var variants = ChooseVariants(y.Symbol, bound, choose, t =>
        {
            var k = Lib("bmm.slang", "bmm", Macros(("M", M), ("N", N), ("K", K), ("NB", nb), ("BSH", IntList(bsh)), ("BSA", IntList(bsa)), ("BSB", IntList(bsb)),
                ("SAM", sam), ("SAK", sak), ("SBK", sbk), ("SBN", sbn), ("ALPHA", alpha), ("M_DYN", mdyn),
                ("BM", t.BM), ("BN", t.BN), ("BK", t.BK), ("TM", t.TM), ("TN", t.TN), ("NT", t.NT)));
            return (k, dyn =>
            {
                long mm = mdyn ? dyn : Mb;
                long batches = nb == 0 ? 1 : (y.Symbol is not null ? dyn * batchesInner : y.Shape.Take(nb).Aggregate(1L, (p, q) => p * q));
                return new Grid(Grid.Div(N, t.BN), Grid.Div(mm, t.BM), (uint)batches);
            }
            );
        });
        KernelArg dynArg = mdyn ? new SymbolArg(y.Symbol!) : new UIntArg(0);
        Emit($"bmm {n.Output.Name} [{M}x{K}x{N}]", [n], variants,
            [Pointer(a), Pointer(b), Pointer(y), dynArg], [a, b], [y]);
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

    private void LowerLayerNorm(Node n)
    {
        var x = RowsInput(n.Inputs[0], "layernorm", out long r1, out long s0, out long s1);
        var w = n.Inputs.Length > 1 ? AsDenseF32(Ref(n.Inputs[1]), "ln-w") : null;
        var bi = n.Inputs.Length > 2 ? AsDenseF32(Ref(n.Inputs[2]), "ln-b") : null;
        var y = Produce(n.Output);
        var k = Lib("rows.slang", "layernorm", Macros(("D", x.Shape[^1]), ("R1", r1), ("S0", s0), ("S1", s1), ("EPS", n.Attributes.Get<float>("eps")),
            ("HAS_W", w is not null), ("HAS_B", bi is not null)));
        var tx = x;
        var wp = w ?? x;
        var bp = bi ?? x;
        Emit($"layernorm {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * r1, 4)))],
            [Pointer(x), Pointer(wp), Pointer(bp), Pointer(y), TokenArg(x)],
            [.. new[] { x, w, bi }.Where(t => t is not null).Select(t => t!)], [y]);
    }

    private void LowerRowReduce(Node n, int mode)
    {
        var x = RowsInput(n.Inputs[0], "rowreduce", out long r1, out long s0, out long s1);
        var y = Produce(n.Output);
        var k = Lib("rows.slang", "rowreduce", Macros(("D", x.Shape[^1]), ("R1", r1), ("S0", s0), ("S1", s1), ("MODE", mode)));
        var tx = x;
        Emit($"{n.Op.Name} {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * r1, 4)))],
            [Pointer(x), Pointer(y), TokenArg(x)], [x], [y]);
    }
}
