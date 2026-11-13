using System.Globalization;
using Sharplight.Runtime;
using System.Text;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    readonly List<object> _temps = [];

    void LowerNode(Node n)
    {
        switch (n.Op.Name)
        {
            case "Reshape": LowerReshape(n); break;
            case "Slice": _refs[n.Output] = ViewSlice(Ref(n.Inputs[0]), n.Attributes); break;
            case "Permute": _refs[n.Output] = ViewPermute(Ref(n.Inputs[0]), n.Attributes.Get<int[]>("perm")); break;
            case "Contiguous": EmitCopy(Ref(n.Inputs[0]), Produce(n.Output), "contiguous", [n]); break;
            case "Concat": LowerConcat(n); break;
            case "MatMul": LowerMatMul(n); break;
            case "MatMulId": LowerMatMulId(n); break;
            case "RmsNorm": LowerNorm(n, mode: 0); break;
            case "L2Norm": LowerNorm(n, mode: 1); break;
            case "Rope": LowerRope(n); break;
            case "Gather": LowerGather(n); break;
            case "CacheWrite": LowerCacheWrite(n); break;
            case "CachedAttention" or "Attention": LowerAttention(n); break;
            case "CausalConv1d": LowerConv(n); break;
            case "GatedDeltaRule": LowerDeltaRule(n); break;
            default:
                if (LowerGeneral(n)) break;
                if (s_elementwise.Contains(n.Op.Name)) { LowerElementwise([n], [n.Output]); break; }
                if (LowerSpatial(n)) break;
                throw new NotSupportedException($"No lowering for op {n.Op.Name}.");
        }
        foreach (var t in _temps) _alloc.Free(t);
        _temps.Clear();
    }

    // ---- views -------------------------------------------------------------------------------
    /// <summary>A reshape of strided data that cannot be expressed as a view (e.g. after a permute) is materialized with a copy.</summary>
    void LowerReshape(Node n)
    {
        var src = Ref(n.Inputs[0]);
        try { _refs[n.Output] = ViewReshape(src, BoundShape(n.Output.Shape), n.Output.Shape.Symbol); }
        catch (NotSupportedException)
        {
            // The output becomes its own root (its memory is not tracked by the view-root liveness of the input; it is
            // simply never reused, which only costs arena space).
            var dst = Produce(n.Output);
            EmitCopy(src, dst with { Shape = src.Shape, Strides = TensorRef.DenseStrides(src.Shape) }, "reshape-copy", [n]);
        }
    }

    static TensorRef ViewSlice(TensorRef r, Attributes a)
    {
        int axis = a.Get<int>("axis");
        if (axis < 0) axis += r.Rank;
        long start = a.Get<long>("start"), len = a.Get<long>("length");
        var shape = (long[])r.Shape.Clone();
        shape[axis] = len;
        return r with { ByteOffset = r.ByteOffset + start * r.Strides[axis] * r.Type.ElementSize, Shape = shape };
    }

    static TensorRef ViewPermute(TensorRef r, int[] perm)
    {
        if (r.Symbol is not null && perm[0] != 0) throw new NotSupportedException("Permute must keep the symbolic axis outermost.");
        return r with { Shape = [.. perm.Select(p => r.Shape[p])], Strides = [.. perm.Select(p => r.Strides[p])] };
    }

    /// <summary>Stride-compatible reshape (PyTorch view semantics).</summary>
    static TensorRef ViewReshape(TensorRef r, long[] newShape, string? symbol)
    {
        var oldShape = r.Shape; var oldStride = r.Strides;
        var newStride = new long[newShape.Length];
        int viewD = newShape.Length - 1;
        long chunkBase = oldStride.Length > 0 ? oldStride[^1] : 1, tensorNumel = 1, viewNumel = 1;
        for (int td = oldShape.Length - 1; td >= 0; td--)
        {
            tensorNumel *= oldShape[td];
            if (td == 0 || (oldShape[td - 1] != 1 && oldStride[td - 1] != tensorNumel * chunkBase))
            {
                while (viewD >= 0 && (viewNumel < tensorNumel || newShape[viewD] == 1))
                {
                    newStride[viewD] = viewNumel * chunkBase;
                    viewNumel *= newShape[viewD];
                    viewD--;
                }
                if (viewNumel != tensorNumel)
                    throw new NotSupportedException($"Reshape [{string.Join(",", oldShape)}] → [{string.Join(",", newShape)}] is not expressible as a view of strides [{string.Join(",", oldStride)}].");
                if (td > 0) { chunkBase = oldStride[td - 1]; tensorNumel = 1; viewNumel = 1; }
            }
        }
        if (viewD != -1) throw new NotSupportedException("Reshape: incompatible shapes.");
        return r with { Shape = newShape, Strides = newStride, Symbol = symbol };
    }

    /// <summary>Rows view of a tensor: [rows = outer*mid, inner] with uniform row stride, inner contiguous.</summary>
    static bool TryRows(TensorRef r, out long rowStride, out long rowsPerOuter)
    {
        rowStride = 0; rowsPerOuter = 1;
        if (r.Rank < 2 || r.Strides[^1] != 1) return false;
        for (int k = 1; k <= r.Rank - 2; k++)
        {
            if (r.Strides[k - 1] != r.Strides[k] * r.Shape[k]) return false;
            rowsPerOuter *= r.Shape[k];
        }
        rowStride = r.Strides[r.Rank - 2];
        return true;
    }

    /// <summary>[T, R1, D] view with strides (S0, S1) for row kernels; middle dims must be collapsible.</summary>
    static bool TryRows3(TensorRef r, out long r1, out long s0, out long s1)
    {
        r1 = 1; s0 = 0; s1 = 0;
        if (r.Strides[^1] != 1) return false;
        if (r.Rank == 1) { s0 = r.Shape[0]; return true; }
        if (r.Rank == 2) { s0 = r.Strides[0]; s1 = r.Shape[1]; return true; }
        for (int k = 2; k <= r.Rank - 2; k++)
            if (r.Shape[k] != 1 && r.Strides[k - 1] != r.Strides[k] * r.Shape[k]) return false;
        for (int k = 1; k <= r.Rank - 2; k++) r1 *= r.Shape[k];
        s0 = r.Strides[0];
        s1 = r.Strides[r.Rank - 2];
        return true;
    }

    /// <summary>Copies a strided tensor into a fresh dense temporary in the arena.</summary>
    TensorRef DenseTemp(TensorRef src, string why)
    {
        var key = new object();
        long off = _alloc.Allocate(key, Math.Max(16, src.Type.ByteSize(src.MaxElements)));
        _temps.Add(key);
        var dst = new TensorRef(_arena, off, src.Type, src.Shape, TensorRef.DenseStrides(src.Shape), src.Symbol);
        EmitCopy(src, dst, "densify:" + why);
        return dst;
    }

    // ---- matmul ------------------------------------------------------------------------------
    /// <summary>Matrix-vector variants for up to 8 rows of x (memory bound: decode, small batches).</summary>
    static List<KernelVariant> MatvecVariants(Func<string, Dictionary<string, string>, CompiledKernel> lib, Dictionary<string, string> m,
        long N, long K, Func<Bindings, long> rows)
    {
        // Decode (one token): split-K matvec, one workgroup per output row, K spread over enough threads that
        // each walks only a few 8-weight groups (keeps many loads in flight per row).
        int skSg = N <= 2048 || K >= 12288 ? 8 : 4;
        var m1 = new Dictionary<string, string>(m) { ["MV_COLS"] = "1", ["MV_SK_SG"] = skSg.ToString(CultureInfo.InvariantCulture) };
        return
        [
            K >= 1024
                ? new KernelVariant(b => rows(b) == 1, lib("matvec_sk", m1), b => new Grid((uint)N))
                : new KernelVariant(b => rows(b) == 1, lib("matvec", m1), b => new Grid(Grid.Div(N, 4))),
            // Small N: one split-K workgroup per row, otherwise one subgroup per row.
            N <= 2048 && K >= 1024
                ? new KernelVariant(b => rows(b) <= 8, lib("matvec_sk", m), b => new Grid((uint)N))
                : new KernelVariant(b => rows(b) <= 8, lib("matvec", m), b => new Grid(Grid.Div(N, 4))),
        ];
    }

    /// <summary>
    /// Two matmuls sharing x whose outputs only feed one elementwise group (e.g. SwiGLU gate/up + silu·mul):
    /// for up to 8 rows one dual matvec computes both and applies the group per element. Larger batches run
    /// the regular launches (matmul, matmul, elementwise), which are skipped for small batches.
    /// </summary>
    void LowerDualMatMul(Node a, Node bNode, List<Node> group, Value output)
    {
        var x = Ref(a.Inputs[0]);
        var wa = Ref(a.Inputs[1]); var wb = Ref(bNode.Inputs[1]);
        long N = wa.Shape[0], K = wa.Shape[1];
        if (!TryRows(x, out long xs, out long mid) || xs % 4 != 0 || x.ByteOffset % 16 != 0 || x.Type != DataType.F32)
        {
            x = DenseTemp(x, "matmul");
            TryRows(x, out xs, out mid);
        }
        var y = Produce(output);
        int fmt = WeightFormat(wa.Type) ?? throw new NotSupportedException($"MatMul weight type {wa.Type}");
        var m = Macros(("WFMT", fmt), ("K", K), ("N", N), ("X_STRIDE", xs), ("Y_STRIDE", N), ("ROW_BYTES", wa.Type.ByteSize(K) + "ull"), ("DUAL", 1));

        var name = new Dictionary<Value, string> { [a.Output] = "m0", [bNode.Output] = "m1" };
        var src = new StringBuilder();
        src.AppendLine("#include \"math_ext.slang\"");
        src.AppendLine("float epi2(float m0, float m1) {");
        foreach (var node in group)
        {
            var vn = "v" + name.Count;
            src.AppendLine($"  float {vn} = {OpExpr(node, [.. node.Inputs.Select(i => name[i])])};");
            name[node.Output] = vn;
        }
        src.AppendLine($"  return {name[output]};");
        src.AppendLine("}");
        src.AppendLine("#include \"matmul.slang\"");
        var source = src.ToString();
        var tx = x; long rowsMul = mid;
        Func<Bindings, long> rows = b => Rows(b, tx) * rowsMul;
        Emit($"matmul2 {output.Name} [{K}→{N}]x2 {wa.Type} fused[{string.Join(",", group.Select(g => g.Op.Name))}]", [a, bNode, .. group],
            MatvecVariants((e, mm) => _compiler.Compile("matmul_dual.slang", source, e, _opt.Target,
                new Dictionary<string, string>(mm) { ["KERNEL_" + e.ToUpperInvariant()] = "1" }, _opt.Cache), m, N, K, rows),
            [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(wa.Buffer, wa.ByteOffset), new PointerArg(wb.Buffer, wb.ByteOffset),
             new PointerArg(y.Buffer, y.ByteOffset), TokenArg(x, mid)],
            [x, wa, wb], [y]);

        var outer = _launchCondition;
        _launchCondition = b => (outer?.Invoke(b) ?? true) && rows(b) > 8;
        LowerMatMul(a);
        LowerMatMul(bNode);
        LowerElementwise(group, [output]);
        _launchCondition = outer;
    }

    static readonly bool s_gemm2 = Environment.GetEnvironmentVariable("SHARPLIGHT_GEMM2") != "0";
    // Tensor-core GEMM layout: warps (M x N) x fragments per warp (M x N); tile = 16*warps*frags.
    // Both layouts produce 128x128 tiles: 4 warps of 64x64 (more workgroups per SM, better for moderate grids) or
    // 8 warps of 64x32 (better when the grid is large). SHARPLIGHT_GEMM2_CFG=wm,wn,fm,fn forces one layout.
    static readonly int[]? s_g2Force = Environment.GetEnvironmentVariable("SHARPLIGHT_GEMM2_CFG")?.Split(',').Select(int.Parse).ToArray();
    const int G2BM = 128, G2BN = 128;
    static Dictionary<string, string> G2Macros(Dictionary<string, string> m, bool large)
    {
        int[] c = s_g2Force ?? (large ? [2, 4, 4, 2] : [2, 2, 4, 4]);
        if (c[0] * c[2] * 16 != G2BM || c[1] * c[3] * 16 != G2BN) throw new NotSupportedException("GEMM layout must give 128x128 tiles.");
        return new(m)
        {
            ["WARPS_M"] = c[0].ToString(CultureInfo.InvariantCulture), ["WARPS_N"] = c[1].ToString(CultureInfo.InvariantCulture),
            ["FRAG_M"] = c[2].ToString(CultureInfo.InvariantCulture), ["FRAG_N"] = c[3].ToString(CultureInfo.InvariantCulture),
            ["GBK"] = s_g2Bk == 64 && long.Parse(m["K"], CultureInfo.InvariantCulture) % 64 == 0 ? "64" : "32",
        };
    }
    static readonly int s_g2Bk = int.Parse(Environment.GetEnvironmentVariable("SHARPLIGHT_GEMM_BK") ?? "64", CultureInfo.InvariantCulture);

    void LowerMatMul(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var wv = n.Inputs[1];
        var w = Ref(wv);
        long N = w.Shape[0], K = w.Shape[1];
        if (K % 8 != 0 && !w.Type.IsQuantized)
        {
            // Weight kernels walk K in groups of 8; odd K goes through the general strided matmul (x · Wᵀ).
            LowerBmm(n, Ref(n.Inputs[0]), w, ta: false, tb: true, alpha: 1f);
            return;
        }
        if (!TryRows(x, out long xs, out long mid) || xs % 4 != 0 || x.ByteOffset % 16 != 0 || x.Type != DataType.F32)
        {
            x = DenseTemp(x, "matmul");
            TryRows(x, out xs, out mid);
        }
        var y = Produce(n.Output);
        int fmt = WeightFormat(w.Type) ?? throw new NotSupportedException($"MatMul weight type {w.Type}");
        var m = Macros(("WFMT", fmt), ("K", K), ("N", N), ("X_STRIDE", xs), ("Y_STRIDE", N), ("ROW_BYTES", w.Type.ByteSize(K) + "ull"));
        long rowsMul = mid;
        var tx = x;
        // Tensor-core path needs f16 matrix MMA (+ f16 arithmetic) and the 32-wide subgroups gemm_cm is written for.
        bool coop = _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16) && _opt.SubgroupSize == 32 && K % 32 == 0
                    && (w.Type != DataType.F32 || _opt.Compile.AllowReducedPrecision);
        Func<Bindings, int> split = b => ChooseSplit(N, K, Rows(b, tx) * rowsMul);
        var big = coop
            ? (s_gemm2 ? new KernelVariant(b => split(b) == 1 && Grid.Div(N, G2BN) * Grid.Div(Rows(b, tx) * rowsMul, G2BM) >= 384,
                             Lib("matmul.slang", "gemm_cm2", G2Macros(m, large: true)), b => new Grid(Grid.Div(N, G2BN), Grid.Div(Rows(b, tx) * rowsMul, G2BM)))
                       : new KernelVariant(_ => true, Lib("matmul.slang", "gemm_cm", m), b => new Grid(Grid.Div(N, 128), Grid.Div(Rows(b, tx) * rowsMul, 64))))
            : new KernelVariant(_ => true, Lib("matmul.slang", "gemm", m), b => new Grid(Grid.Div(N, 64), Grid.Div(Rows(b, tx) * rowsMul, 64)));
        // Tensor-core GEMMs (more than 8 rows) read an f16 copy of x made by a small conversion kernel: half the
        // bytes per tile load (x is re-read by every column tile). Matvecs keep reading f32 x. Separate launches,
        // since a launch's variants share one argument list; the ones not applicable to a binding are skipped.
        bool x16 = coop && s_gemm2 && s_gemmX16;
        if (!x16)
            Emit($"matmul {n.Output.Name} [{K}→{N}] {w.Type}", [n],
                [.. MatvecVariants((e, mm) => Lib("matmul.slang", e, mm), m, N, K, b => Rows(b, tx) * rowsMul), big,
                 .. coop && s_gemm2 ? new[] { new KernelVariant(b => split(b) == 1, Lib("matmul.slang", "gemm_cm2", G2Macros(m, large: false)),
                     b => new Grid(Grid.Div(N, G2BN), Grid.Div(Rows(b, tx) * rowsMul, G2BM))) } : []],
                [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(w.Buffer, w.ByteOffset), new PointerArg(y.Buffer, y.ByteOffset), TokenArg(x, mid)],
                [x, w], [y]);
        else
        {
            Emit($"matmul {n.Output.Name} [{K}→{N}] {w.Type}", [n],
                MatvecVariants((e, mm) => Lib("matmul.slang", e, mm), m, N, K, b => Rows(b, tx) * rowsMul),
                [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(w.Buffer, w.ByteOffset), new PointerArg(y.Buffer, y.ByteOffset), TokenArg(x, mid)],
                [x, w], [y]);
            long rowsMax = tx.Shape[0] * rowsMul;
            var key = new object();
            long off = _alloc.Allocate(key, Math.Max(16, rowsMax * K * 2));
            _temps.Add(key);
            var xh = new TensorRef(_arena, off, DataType.F16, [rowsMax, K], [K, 1], tx.Symbol);
            Emit($"matmul {n.Output.Name} x→f16", [n],
                [new KernelVariant(b => Rows(b, tx) * rowsMul > 8, Lib("data.slang", "to_f16", Macros(("K", K), ("X_STRIDE", xs))),
                    b => new Grid(Grid.Div(Rows(b, tx) * rowsMul * (K / 8), 256)))],
                [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(xh.Buffer, xh.ByteOffset), TokenArg(x, mid)],
                [x], [xh]);
            var mh = new Dictionary<string, string>(m) { ["X_STRIDE"] = K.ToString(CultureInfo.InvariantCulture), ["X_F16"] = "1" };
            Emit($"matmul {n.Output.Name} [{K}→{N}] {w.Type}", [n],
                [new KernelVariant(b => Rows(b, tx) * rowsMul > 8 && split(b) == 1 && Grid.Div(N, G2BN) * Grid.Div(Rows(b, tx) * rowsMul, G2BM) >= 384,
                    Lib("matmul.slang", "gemm_cm2", G2Macros(mh, large: true)), b => new Grid(Grid.Div(N, G2BN), Grid.Div(Rows(b, tx) * rowsMul, G2BM))),
                 new KernelVariant(b => Rows(b, tx) * rowsMul > 8 && split(b) == 1, Lib("matmul.slang", "gemm_cm2", G2Macros(mh, large: false)),
                    b => new Grid(Grid.Div(N, G2BN), Grid.Div(Rows(b, tx) * rowsMul, G2BM)))],
                [new PointerArg(xh.Buffer, xh.ByteOffset), new PointerArg(w.Buffer, w.ByteOffset), new PointerArg(y.Buffer, y.ByteOffset), TokenArg(x, mid)],
                [xh, w], [y]);
            x = xh;
            m = mh;
        }

        // Split-K for grids too small to fill the GPU (small N or few token tiles): partial sums into a shared
        // scratch buffer, then a reduction. Both launches only run when the chosen split is > 1.
        if (coop && s_gemm2)
        {
            long rowsMax = tx.Shape[0] * rowsMul, scratch = 0;
            var splits = new SortedSet<int>();
            // The split depends on rows only through the token tile count; the largest row count per tile count bounds scratch.
            for (long r = Math.Min(rowsMax, 9); r <= rowsMax; r = r < 128 && rowsMax > 128 ? 128 : r == rowsMax ? rowsMax + 1 : Math.Min(rowsMax, r + 128))
            {
                int sp = ChooseSplit(N, K, r);
                if (sp > 1) { splits.Add(sp); scratch = Math.Max(scratch, sp * r * N * 4); }
            }
            if (splits.Count > 0)
            {
                _splitScratch ??= NewBuffer(BufferKind.Activations, "splitk", 0);
                _splitScratch.Bytes = Math.Max(_splitScratch.Bytes, scratch);
                var part = new TensorRef(_splitScratch, 0, DataType.F32, [scratch / 4], [1], null);
                Emit($"matmul {n.Output.Name} [{K}→{N}] {w.Type} split-k", [n],
                    [.. splits.Select(sp => new KernelVariant(b => Rows(b, tx) * rowsMul > 8 && split(b) == sp,
                        Lib("matmul.slang", "gemm_cm2", new Dictionary<string, string>(G2Macros(m, large: false)) { ["SPLITK"] = sp.ToString(CultureInfo.InvariantCulture) }),
                        b => new Grid(Grid.Div(N, G2BN), Grid.Div(Rows(b, tx) * rowsMul, G2BM), (uint)sp)))],
                    [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(w.Buffer, w.ByteOffset), new PointerArg(_splitScratch, 0), TokenArg(x, mid)],
                    [x, w], [part]);
                Emit($"matmul {n.Output.Name} split-k reduce", [n],
                    [.. splits.Select(sp => new KernelVariant(b => Rows(b, tx) * rowsMul > 8 && split(b) == sp,
                        Lib("matmul.slang", "splitk_reduce", new Dictionary<string, string>(m) { ["SPLITK"] = sp.ToString(CultureInfo.InvariantCulture) }),
                        b => new Grid(Grid.Div(Rows(b, tx) * rowsMul * N, 256))))],
                    [new PointerArg(_splitScratch, 0), new PointerArg(w.Buffer, w.ByteOffset), new PointerArg(y.Buffer, y.ByteOffset), TokenArg(x, mid)],
                    [part], [y]);
            }
        }
    }

    BufferDecl? _splitScratch;
    BufferDecl? _attnScratch;
    static readonly bool s_gemmX16 = Environment.GetEnvironmentVariable("SHARPLIGHT_GEMM_X16") != "0";
    static readonly bool s_attnFa = Environment.GetEnvironmentVariable("SHARPLIGHT_ATTN_FA") != "0";
    static readonly bool s_attnSplit = Environment.GetEnvironmentVariable("SHARPLIGHT_ATTN_SPLIT") != "0";

    /// <summary>
    /// K split for the 128x128 tensor-core GEMM: aim for ~256 workgroups, keep at least 4 K blocks per split
    /// and no empty splits.
    /// </summary>
    static int ChooseSplit(long N, long K, long rows)
    {
        if (rows <= 8) return 1;
        long tiles = (N + G2BN - 1) / G2BN * ((rows + G2BM - 1) / G2BM), steps = K / 32;
        if (tiles >= 192) return 1;
        int sp = 1;
        while (sp < 32 && tiles * sp < 256 && steps / (sp * 2) >= 4) sp *= 2;
        while (sp > 1 && (sp - 1) * ((steps + sp - 1) / sp) >= steps) sp /= 2;
        return sp;
    }

    // ---- normalization -----------------------------------------------------------------------
    void LowerNorm(Node n, int mode)
    {
        var x = Ref(n.Inputs[0]);
        if (!TryRows3(x, out long r1, out long s0, out long s1)) { x = DenseTemp(x, "norm"); TryRows3(x, out r1, out s0, out s1); }
        long D = x.Shape[^1];
        TensorRef? w = n.Inputs.Length > 1 ? Ref(n.Inputs[1]) : null;
        if (w is not null && (w.Type != DataType.F32 || !w.IsDense)) throw new NotSupportedException("Norm weight must be dense f32.");
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
            [new PointerArg(x.Buffer, x.ByteOffset), w is null ? new PointerArg(x.Buffer, x.ByteOffset) : new PointerArg(w.Buffer, w.ByteOffset), new PointerArg(y.Buffer, y.ByteOffset), TokenArg(x)],
            w is null ? [x] : [x, w], [y]);
    }

    // ---- rope --------------------------------------------------------------------------------
    /// <param name="cacheWrite">When set, the rotated rows are stored straight into this CacheWrite's cache (no y).</param>
    void LowerRope(Node n, Node? cacheWrite = null)
    {
        // An absorbed RmsNorm of the input is computed in the kernel.
        var norm = _absorbedNorms.GetValueOrDefault(n.Inputs[0]);
        var x = Ref(norm?.Inputs[0] ?? n.Inputs[0]);
        var pos = Ref(n.Inputs[1]);
        if (x.Rank != 3 || x.Strides[2] != 1) x = DenseTemp(x, "rope");
        if (!pos.IsDense) pos = DenseTemp(pos, "rope-pos");
        TensorRef? nw = norm is not null && norm.Inputs.Length > 1 ? Ref(norm.Inputs[1]) : null;
        var y = cacheWrite is null ? Produce(n.Output) : Ref(cacheWrite.Inputs[0]);
        var a = n.Attributes;
        long H = x.Shape[1], D = x.Shape[2];
        int rot = a.Get("dims", (int)D);
        float @base = a.Get<float>("base");
        var sections = a.Get<int[]>("sections", []);
        bool mrope = sections.Length > 0 && sections.Any(s => s > 0);
        var sec = new int[4];
        for (int i = 0; i < Math.Min(4, sections.Length); i++) sec[i] = sections[i];
        long axes = pos.Rank > 1 ? pos.Shape[1] : 1;
        var ff = n.Inputs.Length > 2 ? Ref(n.Inputs[2]) : null;   // optional per-frequency divisors (f32 [rot/2])
        var ffArg = new PointerArg((ff ?? x).Buffer, (ff ?? x).ByteOffset);
        var k = Lib("rope.slang", "main", Macros(("HAS_FF", ff is not null),
            ("H", H), ("D", D), ("S0", x.Strides[0]), ("S1", x.Strides[1]), ("A", axes), ("ROT", rot),
            ("THETA_SCALE", (float)Math.Pow(@base, -2.0 / rot)), ("FREQ_SCALE", a.Get("freq_scale", 1f)),
            ("NEOX", a.Get("style", "neox") == "neox"), ("MROPE", mrope), ("INTERLEAVED", a.Get("interleaved", false)),
            ("SEC0", sec[0]), ("SEC1", sec[1]), ("SEC2", sec[2]), ("SEC3", sec[3])));
        var tx = x;
        if (norm is not null || cacheWrite is not null)
        {
            if (norm is not null && nw is null) throw new NotSupportedException("Fused rope norm expects a weight.");
            if (cacheWrite is not null && y.Type != DataType.F16 && y.Type != DataType.F32) throw new NotSupportedException($"Cache type {y.Type}");
            var rm = Macros(("HAS_FF", ff is not null),
                ("H", H), ("D", D), ("S0", x.Strides[0]), ("S1", x.Strides[1]), ("A", axes), ("ROT", rot),
                ("THETA_SCALE", (float)Math.Pow(@base, -2.0 / rot)), ("FREQ_SCALE", a.Get("freq_scale", 1f)),
                ("NEOX", a.Get("style", "neox") == "neox"), ("MROPE", mrope), ("INTERLEAVED", a.Get("interleaved", false)),
                ("SEC0", sec[0]), ("SEC1", sec[1]), ("SEC2", sec[2]), ("SEC3", sec[3]),
                ("NORM", norm is not null), ("EPS", norm?.Attributes.Get<float>("eps") ?? 0f),
                ("CACHE_OUT", cacheWrite is not null), ("CACHE_F16", y.Type == DataType.F16), ("RT_N_PAST", cacheWrite is null ? 0 : Slot(cacheWrite)));
            var rk = Lib("rope.slang", "rope_row", rm);
            var wRef = nw ?? x;
            Emit($"rope{(norm is null ? "" : "+norm")}{(cacheWrite is null ? "" : "+cache_write")} {n.Output.Name}", [.. new[] { norm, n, cacheWrite }.OfType<Node>()],
                [Always(rk, b => new Grid((uint)(Rows(b, tx) * H)))],
                [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(pos.Buffer, pos.ByteOffset), new PointerArg(wRef.Buffer, wRef.ByteOffset),
                 new PointerArg(y.Buffer, y.ByteOffset), new PointerArg(_runtime, 0), ffArg, TokenArg(x)],
                [.. (nw is null ? [x, pos] : (TensorRef[])[x, pos, nw]), .. ff is null ? [] : (TensorRef[])[ff]], [y]);
            return;
        }
        Emit($"rope {n.Output.Name}", [n],
            [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * H * (D / 2), 64)))],
            [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(pos.Buffer, pos.ByteOffset), new PointerArg(y.Buffer, y.ByteOffset), ffArg, TokenArg(x)],
            ff is null ? [x, pos] : [x, pos, ff], [y]);
    }

    // ---- gather ------------------------------------------------------------------------------
    void LowerGather(Node n)
    {
        var table = Ref(n.Inputs[0]);
        var idx = Ref(n.Inputs[1]);
        if (!idx.IsDense) idx = DenseTemp(idx, "gather-idx");
        long D = table.Shape[^1];
        long rowBytes;
        if (table.Type.IsQuantized || table.IsDense) rowBytes = table.Type.ByteSize(D);
        else
        {
            if (!TryRows(table, out long rs, out _) || (rs * 4) % 16 != 0 || table.ByteOffset % 16 != 0) { table = DenseTemp(table, "gather"); rs = D; }
            rowBytes = rs * table.Type.ElementSize;
        }
        if (n.Output.Type != DataType.F32) throw new NotSupportedException($"Gather to {n.Output.Type} (only f32 output is implemented).");
        var y = Produce(n.Output);
        int fmt = WeightFormat(table.Type) ?? throw new NotSupportedException($"Gather table type {table.Type}");
        var k = Lib("data.slang", "gather", Macros(("D", D), ("WFMT", fmt), ("ROW_BYTES", rowBytes + "ull")));
        var ti = idx;
        Emit($"gather {n.Output.Name}", [n],
            [Always(k, b => new Grid(Grid.Div(Rows(b, ti) * (D / 8), 128)))],
            [new PointerArg(table.Buffer, table.ByteOffset), new PointerArg(idx.Buffer, idx.ByteOffset), new PointerArg(y.Buffer, y.ByteOffset), TokenArg(idx)],
            [table, idx], [y]);
    }

    // ---- caches & attention -------------------------------------------------------------------
    int Slot(Node n) => _scalarSlots[n.Attributes.Get("offset", WellKnown.PastScalar)];

    void LowerCacheWrite(Node n)
    {
        if (_absorbedRope.TryGetValue(n.Inputs[1], out var rope)) { LowerRope(rope, cacheWrite: n); return; }
        var state = Ref(n.Inputs[0]);
        var x = Ref(n.Inputs[1]);
        if (!TryRows3(x, out long r1, out long s0, out long s1)) { x = DenseTemp(x, "cache"); TryRows3(x, out r1, out s0, out s1); }
        long inner = x.Shape[^1];
        bool f16 = state.Type == DataType.F16;
        if (!f16 && state.Type != DataType.F32) throw new NotSupportedException($"Cache type {state.Type}");
        var k = Lib("data.slang", "cache_write", Macros(("R1", r1), ("INNER", inner), ("S0", s0), ("S1", s1), ("DST_F16", f16), ("RT_N_PAST", Slot(n))));
        var tx = x;
        Emit($"cache_write {state.Buffer.Name}", [n],
            [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * r1 * inner, 128)))],
            [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(state.Buffer, state.ByteOffset), new PointerArg(_runtime, 0), TokenArg(x)],
            [x], [state]);
    }

    void LowerAttention(Node n)
    {
        // "Attention": bidirectional self-attention over K/V given as plain tensors (vision encoders), every query sees all T keys.
        bool full = n.Op.Name == "Attention";
        var q = Ref(n.Inputs[0]);
        var kc = Ref(n.Inputs[1]);
        var vc = Ref(n.Inputs[2]);
        if (q.Rank != 3 || q.Strides[2] != 1) q = DenseTemp(q, "attn-q");
        if (full)
        {
            kc = AsDenseF32(kc, "attn-k");
            vc = AsDenseF32(vc, "attn-v");
        }
        var o = Produce(n.Output);
        long HQ = q.Shape[1], D = q.Shape[2], HKV = kc.Shape[1], DV = vc.Shape[2];
        if (kc.Type != vc.Type) throw new NotSupportedException("K and V caches must have the same type.");
        int window = n.Attributes.Get("window", 0);
        float softcap = n.Attributes.Get("softcap", 0f);
        var m = Macros(("HQ", HQ), ("HKV", HKV), ("D", D), ("DV", DV), ("QS0", q.Strides[0]), ("QS1", q.Strides[1]),
            ("SCALE", n.Attributes.Get("scale", 1f)), ("KV_F16", kc.Type == DataType.F16), ("FULL_ATTN", full), ("RT_N_PAST", full ? 0 : Slot(n)),
            ("WINDOW", window), ("HAS_SOFTCAP", softcap > 0), ("SOFTCAP", softcap > 0 ? softcap : 1f));
        var k = Lib("attention.slang", "main", m);
        var tq = q;
        // Few queries (decode): one workgroup per (head, query) leaves most of the GPU idle, so the keys are split
        // across SPLITS workgroups per head (flash decoding) and merged by a small combine kernel.
        const long SplitMaxRows = 4;
        int splits = (int)Math.Clamp((256 + HQ - 1) / HQ, 1, 32);
        bool split = !full && splits > 1 && s_attnSplit;
        long rowsMax = tq.Shape[0];
        var part = o;
        if (split)
        {
            long bytes = splits * Math.Min(rowsMax, SplitMaxRows) * HQ * (DV + 2) * 4;
            _attnScratch ??= NewBuffer(BufferKind.Activations, "attn-split", 0);
            _attnScratch.Bytes = Math.Max(_attnScratch.Bytes, bytes);
            part = new TensorRef(_attnScratch, 0, DataType.F32, [bytes / 4], [1], null);
        }
        var sm = new Dictionary<string, string>(m) { ["SPLITS"] = splits.ToString(CultureInfo.InvariantCulture) };
        // Many queries (prefill): tensor-core flash attention when the device and shapes allow it.
        bool fa = !full && s_attnFa && _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16) && _opt.Has(DeviceFeatures.SubgroupOps)
                  && _opt.SubgroupSize == 32 && kc.Type == DataType.F16 && D % 16 == 0 && DV % 64 == 0 && D <= 256 && DV <= 256 && HQ % HKV == 0
                  && window == 0 && softcap == 0;   // the flash kernel has no sliding window / soft cap
        var variants = new List<KernelVariant>();
        if (split) variants.Add(new KernelVariant(b => Rows(b, tq) <= SplitMaxRows, Lib("attention.slang", "main", sm), b => new Grid((uint)HQ, (uint)Rows(b, tq), (uint)splits)));
        if (fa) variants.Add(new KernelVariant(b => Rows(b, tq) > SplitMaxRows, Lib("attention.slang", "fa", m), b => new Grid((uint)HQ, Grid.Div(Rows(b, tq), 32))));
        variants.Add(Always(k, b => new Grid((uint)HQ, (uint)Rows(b, tq))));
        Emit($"attention {n.Output.Name}", [n],
            variants,
            [new PointerArg(q.Buffer, q.ByteOffset), new PointerArg(kc.Buffer, kc.ByteOffset), new PointerArg(vc.Buffer, vc.ByteOffset),
             new PointerArg(o.Buffer, o.ByteOffset), new PointerArg(part.Buffer, part.ByteOffset), new PointerArg(_runtime, 0), TokenArg(q)],
            [q, kc, vc], split ? [o, part] : [o]);
        if (split)
            Emit($"attention {n.Output.Name} combine", [n],
                [new KernelVariant(b => Rows(b, tq) <= SplitMaxRows, Lib("attention.slang", "combine", sm), b => new Grid((uint)HQ, (uint)Rows(b, tq)))],
                [new PointerArg(part.Buffer, part.ByteOffset), new PointerArg(o.Buffer, o.ByteOffset), TokenArg(q)],
                [part], [o]);
    }

    // ---- recurrent ---------------------------------------------------------------------------
    void LowerConv(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var w = Ref(n.Inputs[1]);
        var st = Ref(n.Inputs[2]);
        if (x.Rank != 2 || x.Strides[1] != 1) x = DenseTemp(x, "conv");
        if (w.Type != DataType.F32) throw new NotSupportedException("conv weight must be f32");
        var y = Produce(n.Output);
        long C = x.Shape[1], KS = w.Shape[1];
        var k = Lib("recurrent.slang", "conv1d", Macros(("C", C), ("KS", KS), ("XS", x.Strides[0]), ("ACT_SILU", n.Attributes.Get("activation", "none") == "silu")));
        Emit($"conv1d {n.Output.Name}", [n],
            [Always(k, _ => new Grid(Grid.Div(C, 128)))],
            [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(w.Buffer, w.ByteOffset), new PointerArg(st.Buffer, st.ByteOffset), new PointerArg(y.Buffer, y.ByteOffset), TokenArg(x)],
            [x, w, st], [y, st]);
    }

    void LowerDeltaRule(Node n)
    {
        // q/k may come straight from an absorbed L2Norm: read its input and normalize in the kernel.
        Node? nq = _absorbedNorms.GetValueOrDefault(n.Inputs[0]), nk = _absorbedNorms.GetValueOrDefault(n.Inputs[1]);
        var q = Ref(nq?.Inputs[0] ?? n.Inputs[0]); var k = Ref(nk?.Inputs[0] ?? n.Inputs[1]); var v = Ref(n.Inputs[2]);
        var g = Ref(n.Inputs[3]); var beta = Ref(n.Inputs[4]); var st = Ref(n.Inputs[5]);
        if (q.Strides[2] != 1) q = DenseTemp(q, "delta-q");
        if (k.Strides[2] != 1) k = DenseTemp(k, "delta-k");
        if (v.Strides[2] != 1) v = DenseTemp(v, "delta-v");
        if (!g.IsDense) g = DenseTemp(g, "delta-g");
        if (!beta.IsDense) beta = DenseTemp(beta, "delta-beta");
        var o = Produce(n.Output);
        long HK = q.Shape[1], DK = q.Shape[2], HV = v.Shape[1], DV = v.Shape[2];
        int lpr = int.Parse(Environment.GetEnvironmentVariable("SHARPLIGHT_DELTA_LPR") ?? "8");
        lpr = Math.Min(lpr, _opt.SubgroupSize);
        while ((DV * lpr > 1024 || DK % lpr != 0) && lpr > 1) lpr /= 2;
        // Split each head's DV state rows over several workgroups (rows are independent) for occupancy.
        long dvbWant = long.Parse(Environment.GetEnvironmentVariable("SHARPLIGHT_DELTA_DVB") ?? "16");
        long dvb = DV % dvbWant == 0 ? dvbWant : DV;
        var kern = Lib("recurrent.slang", "delta_rule", Macros(("HK", HK), ("HV", HV), ("DK", DK), ("DV", DV),
            ("QS0", q.Strides[0]), ("QS1", q.Strides[1]), ("KS0", k.Strides[0]), ("KS1", k.Strides[1]), ("VS0", v.Strides[0]), ("VS1", v.Strides[1]),
            ("SCALE", n.Attributes.Get<float>("scale")), ("LPR", lpr), ("DVB", dvb), ("TB", Math.Clamp(2048 / DK, 1, 16)),
            ("NORM_Q", nq is not null), ("EPS_Q", nq?.Attributes.Get<float>("eps") ?? 0f), ("NORM_K", nk is not null), ("EPS_K", nk?.Attributes.Get<float>("eps") ?? 0f)));
        Emit($"delta_rule {n.Output.Name}", [.. new[] { nq, nk }.OfType<Node>(), n],
            [Always(kern, _ => new Grid((uint)HV, (uint)(DV / dvb)))],
            [new PointerArg(q.Buffer, q.ByteOffset), new PointerArg(k.Buffer, k.ByteOffset), new PointerArg(v.Buffer, v.ByteOffset),
             new PointerArg(g.Buffer, g.ByteOffset), new PointerArg(beta.Buffer, beta.ByteOffset), new PointerArg(st.Buffer, st.ByteOffset),
             new PointerArg(o.Buffer, o.ByteOffset), TokenArg(q)],
            [q, k, v, g, beta, st], [o, st]);
    }

    // ---- concat ------------------------------------------------------------------------------
    void LowerConcat(Node n)
    {
        var y = Produce(n.Output);
        int axis = n.Attributes.Get<int>("axis");
        if (axis < 0) axis += y.Rank;
        long start = 0;
        foreach (var i in n.Inputs)
        {
            var src = Ref(i);
            var shape = (long[])y.Shape.Clone();
            shape[axis] = src.Shape[axis];
            var dst = y with { ByteOffset = y.ByteOffset + start * y.Strides[axis] * y.Type.ElementSize, Shape = shape };
            EmitCopy(src, dst, $"concat {n.Output.Name}", [n]);
            start += src.Shape[axis];
        }
    }

    // ---- reset -------------------------------------------------------------------------------
    List<KernelLaunch> BuildResetLaunches()
    {
        var list = new List<KernelLaunch>();
        if (_stateBuffers.Count == 0) return list;
        var k = Lib("data.slang", "fill", Macros());
        foreach (var b in _stateBuffers.Values)
        {
            uint count = (uint)(b.Bytes / 4);
            uint groups = Math.Min(65535u, Grid.Div(count, 256));
            list.Add(new KernelLaunch
            {
                Name = "reset " + b.Name,
                Variants = [Always(k, _ => new Grid(groups))],
                Args = [new PointerArg(b, 0), new UIntArg(count), new UIntArg(0), new UIntArg(groups * 256)],
                Reads = [],
                Writes = [new BufferRange(b, 0, b.Bytes)],
            });
        }
        return list;
    }

    // ---- elementwise codegen -----------------------------------------------------------------
    void EmitCopy(TensorRef src, TensorRef dst, string name, IEnumerable<Node>? nodes = null)
    {
        if (src.Type == dst.Type && src.Type != DataType.F32) { EmitRawCopy(src, dst, name, nodes ?? []); return; }
        if (dst.Type != DataType.F32) throw new NotSupportedException($"copy {src.Type} → {dst.Type}");
        EmitElementwiseKernel(name, nodes ?? [], dst.Shape, dst.Symbol, [src], [(dst, "a0")], []);
    }

    void LowerElementwise(List<Node> group, List<Value> outputs, Node? normTail = null, Node? normHead = null)
    {
        var shapeV = group[0].Output.Shape;
        var inputs = new List<TensorRef>();
        var inputIndex = new Dictionary<Value, int>();
        var produced = group.Select(g => g.Output).ToHashSet();
        var stmts = new List<string>();
        var name = new Dictionary<Value, string>();

        string Operand(Value v)
        {
            if (name.TryGetValue(v, out var s)) return s;
            if (!inputIndex.TryGetValue(v, out int idx))
            {
                idx = inputs.Count;
                inputs.Add(Ref(v));
                inputIndex[v] = idx;
            }
            return name[v] = "a" + idx;
        }

        string? headInput = null;
        if (normHead is not null)
        {
            headInput = Operand(normHead.Inputs[0]);
            name[normHead.Output] = "nrm";
        }
        foreach (var node in group)
        {
            var args = node.Inputs.Select(Operand).ToArray();
            string expr = OpExpr(node, args);
            var vn = "v" + stmts.Count;
            stmts.Add($"float {vn} = {expr};");
            name[node.Output] = vn;
        }

        var outs = outputs.Select(o => (Produce(o), name[o])).ToList();
        var shape = BoundShape(shapeV);
        var label = string.Join(",", group.Select(g => g.Op.Name));
        if (normTail is not null)
        {
            TensorRef? w = normTail.Inputs.Length > 1 ? Ref(normTail.Inputs[1]) : null;
            EmitRowNormKernel($"fused[{label}]+{normTail.Op.Name}", [.. group, normTail], shape, shapeV.Symbol,
                inputs, preStmts: stmts, normExpr: name[normTail.Inputs[0]], pass1Outputs: outs, postStmts: [],
                pass2Outputs: [(Produce(normTail.Output), "nrm")], w, normTail.Op.Name == "RmsNorm" ? 0 : 1, normTail.Attributes.Get<float>("eps"));
            return;
        }
        if (normHead is not null)
        {
            TensorRef? w = normHead.Inputs.Length > 1 ? Ref(normHead.Inputs[1]) : null;
            EmitRowNormKernel($"{normHead.Op.Name}+fused[{label}]", [normHead, .. group], shape, shapeV.Symbol,
                inputs, preStmts: [], normExpr: headInput!, pass1Outputs: [], postStmts: stmts,
                pass2Outputs: outs, w, normHead.Op.Name == "RmsNorm" ? 0 : 1, normHead.Attributes.Get<float>("eps"));
            return;
        }
        EmitElementwiseKernel($"fused[{string.Join(",", group.Select(g => g.Op.Name))}]", group, shape, shapeV.Symbol, inputs, outs, stmts);
    }

    /// <summary>
    /// Row normalization over the innermost axis fused with elementwise work before it (computing the normalized
    /// value) and/or after it (consuming <c>nrm</c>), one workgroup per row. Pass 1 evaluates the pre statements,
    /// writes <paramref name="pass1Outputs"/> and reduces the sum of squares; pass 2 re-evaluates the pre statements
    /// (cheaper than keeping a row in shared memory), forms <c>nrm</c>, runs the post statements and writes
    /// <paramref name="pass2Outputs"/>.
    /// </summary>
    void EmitRowNormKernel(string name, IEnumerable<Node> nodes, long[] shape, string? symbol, List<TensorRef> inputs,
        List<string> preStmts, string normExpr, List<(TensorRef Ref, string Expr)> pass1Outputs, List<string> postStmts,
        List<(TensorRef Ref, string Expr)> pass2Outputs, TensorRef? w, int mode, float eps)
    {
        int rank = shape.Length;
        var P = new long[rank];
        long acc = 1;
        for (int k = rank - 1; k >= 0; k--) { P[k] = acc; acc *= shape[k]; }
        long D = shape[^1], inner = rank == 0 ? 1 : P[0];
        const int threads = 256;
        var outputs = pass1Outputs.Concat(pass2Outputs).ToList();

        var sb = new StringBuilder();
        sb.AppendLine("#include \"math_ext.slang\"");
        sb.AppendLine("struct Params {");
        for (int i = 0; i < inputs.Count; i++) sb.AppendLine($"  {ElemPtrType(inputs[i].Type)}* in{i};");
        sb.AppendLine("  float* w;");
        for (int i = 0; i < outputs.Count; i++) sb.AppendLine($"  float* out{i};");
        sb.AppendLine("  uint rows;");
        sb.AppendLine("};");
        sb.AppendLine($"groupshared float red[{threads}];");
        sb.AppendLine($"[shader(\"compute\")] [numthreads({threads}, 1, 1)]");
        sb.AppendLine("void main(uint3 gid : SV_GroupID, uint3 lt : SV_GroupThreadID, uniform Params p) {");
        sb.AppendLine("  uint row = gid.x; if (row >= p.rows) return;");
        // Fixed trip count per thread. Short rows (decode: one workgroup per row, latency bound) are fully
        // unrolled with every load issued before any store: inputs and outputs are plain pointers that may alias, so
        // an interleaved load/store loop would serialize one memory round trip per element. The loaded values stay in
        // registers for the second pass.
        long iters = (D + threads - 1) / threads;
        bool unroll = iters <= 32;
        void Indices(string indent)
        {
            sb.AppendLine($"{indent}uint idx = row * {D}u + i;");
            for (int k = 0; k < rank; k++)
                sb.AppendLine(k == 0 ? $"{indent}uint d0 = idx / {P[0]}u;" : $"{indent}uint d{k} = (idx / {P[k]}u) % {shape[k]}u;");
        }
        void Loop(string body)
        {
            sb.AppendLine($"  [ForceUnroll] for (uint j = 0; j < {iters}u; j++) {{");
            sb.AppendLine($"    uint i = lt.x + j * {threads}u;");
            if (D % threads != 0) sb.AppendLine($"    if (i < {D}u) {{");
            sb.Append(body);
            if (D % threads != 0) sb.AppendLine("    }");
            sb.AppendLine("  }");
        }
        void Body(bool second)
        {
            if (unroll)
            {
                var inner = new StringBuilder();
                var save = sb; sb = inner;
                Indices("    ");
                for (int i = 0; i < inputs.Count; i++) sb.AppendLine($"    float a{i} = ar{i}[j];");
                foreach (var st in preStmts) sb.AppendLine("    " + st);
                if (!second)
                {
                    for (int i = 0; i < pass1Outputs.Count; i++)
                        sb.AppendLine($"    p.out{i}[{ElemOffset(pass1Outputs[i].Ref, shape)}] = {pass1Outputs[i].Expr};");
                    sb.AppendLine($"    ss += {normExpr} * {normExpr};");
                }
                else
                {
                    sb.AppendLine($"    float nrm = {normExpr} * scale{(w is null ? "" : " * wr[j]")};");
                    foreach (var st in postStmts) sb.AppendLine("    " + st);
                    for (int i = 0; i < pass2Outputs.Count; i++)
                        sb.AppendLine($"    p.out{pass1Outputs.Count + i}[{ElemOffset(pass2Outputs[i].Ref, shape)}] = {pass2Outputs[i].Expr};");
                }
                sb = save;
                if (!second)
                {
                    for (int i = 0; i < inputs.Count; i++) sb.AppendLine($"  float ar{i}[{iters}];");
                    var loads = new StringBuilder();
                    var save2 = sb; sb = loads;
                    Indices("    ");
                    for (int i = 0; i < inputs.Count; i++) sb.AppendLine($"    ar{i}[j] = {ElemLoad(inputs[i], i, shape)};");
                    if (w is not null) sb.AppendLine("    wr[j] = p.w[i];");
                    sb = save2;
                    if (w is not null) sb.AppendLine($"  float wr[{iters}];");
                    Loop(loads.ToString());
                }
                Loop(inner.ToString());
                return;
            }
            sb.AppendLine($"  for (uint i = lt.x; i < {D}u; i += {threads}u) {{");
            Indices("    ");
            for (int i = 0; i < inputs.Count; i++)
                sb.AppendLine($"    float a{i} = {ElemLoad(inputs[i], i, shape)};");
            foreach (var st in preStmts) sb.AppendLine("    " + st);
            if (!second)
            {
                for (int i = 0; i < pass1Outputs.Count; i++)
                    sb.AppendLine($"    p.out{i}[{ElemOffset(pass1Outputs[i].Ref, shape)}] = {pass1Outputs[i].Expr};");
                sb.AppendLine($"    ss += {normExpr} * {normExpr};");
            }
            else
            {
                sb.AppendLine($"    float nrm = {normExpr} * scale{(w is null ? "" : " * p.w[i]")};");
                foreach (var st in postStmts) sb.AppendLine("    " + st);
                for (int i = 0; i < pass2Outputs.Count; i++)
                    sb.AppendLine($"    p.out{pass1Outputs.Count + i}[{ElemOffset(pass2Outputs[i].Ref, shape)}] = {pass2Outputs[i].Expr};");
            }
            sb.AppendLine("  }");
        }
        sb.AppendLine("  float ss = 0;");
        Body(false);
        if (_opt.Has(DeviceFeatures.SubgroupOps))
        {
            // Subgroup sums, then one barrier and a short serial sum over the subgroups.
            sb.AppendLine("  float ws = WaveActiveSum(ss);");
            sb.AppendLine("  uint lanes = WaveGetLaneCount();");
            sb.AppendLine("  if (WaveIsFirstLane()) red[lt.x / lanes] = ws;");
            sb.AppendLine("  GroupMemoryBarrierWithGroupSync();");
            sb.AppendLine($"  float tot = 0; for (uint k = 0; k < {threads}u / lanes; k++) tot += red[k];");
        }
        else
        {
            sb.AppendLine("  red[lt.x] = ss;");
            sb.AppendLine("  GroupMemoryBarrierWithGroupSync();");
            sb.AppendLine($"  for (uint s = {threads / 2}u; s > 0; s >>= 1) {{ if (lt.x < s) red[lt.x] += red[lt.x + s]; GroupMemoryBarrierWithGroupSync(); }}");
            sb.AppendLine("  float tot = red[0];");
        }
        sb.AppendLine(mode == 0 ? $"  float scale = rsqrt(tot / {F(D)} + {F(eps)});" : $"  float scale = rsqrt(tot + {F(eps)});");
        Body(true);
        sb.AppendLine("}");

        var kernel = _compiler.Compile("elementwise.slang", sb.ToString(), "main", _opt.Target, Macros(), _opt.Cache);
        long rowsPerSym = inner / D;
        KernelArg rows = symbol is null ? new UIntArg((uint)(shape.Aggregate(1L, (a, b) => a * b) / D)) : new SymbolArg(symbol, rowsPerSym);
        var wArg = w ?? outputs[0].Ref;
        Emit(name, nodes,
            [Always(kernel, b => new Grid((uint)((symbol is null ? shape[0] : b[symbol]) * rowsPerSym)))],
            [.. inputs.Select(r => (KernelArg)new PointerArg(r.Buffer, r.ByteOffset)), new PointerArg(wArg.Buffer, wArg.ByteOffset),
             .. outputs.Select(o => (KernelArg)new PointerArg(o.Ref.Buffer, o.Ref.ByteOffset)), rows],
            w is null ? inputs : [.. inputs, w], outputs.Select(o => o.Ref));
    }

    static string ElemOffset(TensorRef r, long[] shape)
    {
        int rank = shape.Length;
        var terms = new List<string>();
        for (int k = 0; k < rank; k++)
        {
            int rk = r.Rank - rank + k;
            if (rk < 0) continue;
            long st = r.Shape[rk] == 1 && shape[k] != 1 ? 0 : r.Strides[rk];
            if (st == 0 || (shape[k] == 1)) continue;
            terms.Add(st == 1 ? $"d{k}" : $"d{k} * {st}u");
        }
        return terms.Count == 0 ? "0" : string.Join(" + ", terms);
    }

    static string ElemLoad(TensorRef r, int i, long[] shape) => r.Type == DataType.F32 ? $"p.in{i}[{ElemOffset(r, shape)}]"
        : r.Type == DataType.F16 ? $"h2f(p.in{i}[{ElemOffset(r, shape)}])"
        : r.Type == DataType.BF16 ? $"bf2f(p.in{i}[{ElemOffset(r, shape)}])"
        : r.Type == DataType.I32 || r.Type == DataType.I64 || r.Type == DataType.U32 ? $"float(p.in{i}[{ElemOffset(r, shape)}])"
        : throw new NotSupportedException($"elementwise input type {r.Type}");

    static string ElemPtrType(DataType t) => t == DataType.F32 ? "float" : t == DataType.I32 ? "int" : t == DataType.I64 ? "int64_t" : t == DataType.U32 ? "uint" : "uint16_t";

    /// <summary>Slang expression of an elementwise node applied to operand expressions.</summary>
    static string OpExpr(Node node, string[] args)
    {
    return node.Op.Name switch
        {
            "Neg" => $"(-{args[0]})",
            "Exp" => $"exp({args[0]})",
            "Log" => $"log({args[0]})",
            "Sqrt" => $"sqrt({args[0]})",
            "Rsqrt" => $"rsqrt({args[0]})",
            "Sigmoid" => $"(1.0 / (1.0 + exp(-{args[0]})))",
            "Silu" => $"({args[0]} / (1.0 + exp(-{args[0]})))",
            "Gelu" => $"gelu_erf({args[0]})",
            "GeluTanh" => $"gelu_tanh({args[0]})",
            "Tanh" => $"tanh({args[0]})",
            "Softplus" => $"({args[0]} > 20.0 ? {args[0]} : log(1.0 + exp({args[0]})))",
            "Relu" => $"max({args[0]}, 0.0)",
            "Abs" => $"abs({args[0]})",
            "Square" => $"({args[0]} * {args[0]})",
            "Recip" => $"(1.0 / {args[0]})",
            "Add" => $"({args[0]} + {args[1]})",
            "Sub" => $"({args[0]} - {args[1]})",
            "Mul" => $"({args[0]} * {args[1]})",
            "Div" => $"({args[0]} / {args[1]})",
            "Max" => $"max({args[0]}, {args[1]})",
            "Min" => $"min({args[0]}, {args[1]})",
            "Scale" => $"({args[0]} * {F(node.Attributes.Get("scale", 1f))} + {F(node.Attributes.Get("bias", 0f))})",
            "Cast" => args[0],
            "Sin" => $"sin({args[0]})", "Cos" => $"cos({args[0]})", "Tan" => $"tan({args[0]})",
            "Asin" => $"asin({args[0]})", "Acos" => $"acos({args[0]})", "Atan" => $"atan({args[0]})",
            "Sinh" => $"sinh({args[0]})", "Cosh" => $"cosh({args[0]})",
            "Asinh" => $"asinh_f({args[0]})",
            "Acosh" => $"log({args[0]} + sqrt({args[0]} * {args[0]} - 1.0))",
            "Atanh" => $"(0.5 * log((1.0 + {args[0]}) / (1.0 - {args[0]})))",
            "Floor" => $"floor({args[0]})", "Ceil" => $"ceil({args[0]})", "Round" => $"round_even({args[0]})", "Sign" => $"sign_f({args[0]})",
            "Softsign" => $"({args[0]} / (1.0 + abs({args[0]})))",
            "Mish" => $"({args[0]} * tanh({args[0]} > 20.0 ? {args[0]} : log(1.0 + exp({args[0]}))))",
            "Elu" => $"({args[0]} > 0.0 ? {args[0]} : {F(node.Attributes.Get("alpha", 1f))} * (exp({args[0]}) - 1.0))",
            "Celu" => $"(max({args[0]}, 0.0) + min(0.0, {F(node.Attributes.Get("alpha", 1f))} * (exp({args[0]} / {F(node.Attributes.Get("alpha", 1f))}) - 1.0)))",
            "Selu" => $"({F(node.Attributes.Get("gamma", 1.0507009873554805f))} * ({args[0]} > 0.0 ? {args[0]} : {F(node.Attributes.Get("alpha", 1.6732632423543772f))} * (exp({args[0]}) - 1.0)))",
            "LeakyRelu" => $"({args[0]} >= 0.0 ? {args[0]} : {F(node.Attributes.Get("alpha", 0.01f))} * {args[0]})",
            "ThresholdedRelu" => $"({args[0]} > {F(node.Attributes.Get("alpha", 1f))} ? {args[0]} : 0.0)",
            "Not" => $"({args[0]} == 0.0 ? 1.0 : 0.0)",
            "IsNaN" => $"(isnan({args[0]}) ? 1.0 : 0.0)",
            "IsInf" => $"(isinf({args[0]}) ? 1.0 : 0.0)",
            "Mod" => node.Attributes.Get("fmod", false) ? $"fmod({args[0]}, {args[1]})" : $"mod_py({args[0]}, {args[1]})",
            "PRelu" => $"({args[0]} >= 0.0 ? {args[0]} : {args[1]} * {args[0]})",
            "Equal" => $"({args[0]} == {args[1]} ? 1.0 : 0.0)", "NotEqual" => $"({args[0]} != {args[1]} ? 1.0 : 0.0)",
            "Less" => $"({args[0]} < {args[1]} ? 1.0 : 0.0)", "LessEqual" => $"({args[0]} <= {args[1]} ? 1.0 : 0.0)",
            "Greater" => $"({args[0]} > {args[1]} ? 1.0 : 0.0)", "GreaterEqual" => $"({args[0]} >= {args[1]} ? 1.0 : 0.0)",
            "And" => $"(({args[0]} != 0.0 && {args[1]} != 0.0) ? 1.0 : 0.0)", "Or" => $"(({args[0]} != 0.0 || {args[1]} != 0.0) ? 1.0 : 0.0)",
            "Xor" => $"(({args[0]} != 0.0) != ({args[1]} != 0.0) ? 1.0 : 0.0)",
            "Where" => $"({args[0]} != 0.0 ? {args[1]} : {args[2]})",
            "Clip" => ClipExpr(args[0], node.Attributes.Get("min", float.NegativeInfinity), node.Attributes.Get("max", float.PositiveInfinity)),
            "Erf" => $"erf_accurate({args[0]})",
            "HardSwish" => $"({args[0]} * clamp({args[0]} * {F(1f / 6f)} + 0.5, 0.0, 1.0))",
            "HardSigmoid" => $"clamp({args[0]} * {F(node.Attributes.Get("alpha", 0.2f))} + {F(node.Attributes.Get("beta", 0.5f))}, 0.0, 1.0)",
            "Pow" => $"pow_onnx({args[0]}, {args[1]})",
            _ => throw new NotSupportedException(node.Op.Name),
        };
    }

    static string ClipExpr(string x, float lo, float hi)
    {
        var e = x;
        if (!float.IsNegativeInfinity(lo)) e = $"max({e}, {F(lo)})";
        if (!float.IsPositiveInfinity(hi)) e = $"min({e}, {F(hi)})";
        return $"({e})";
    }

    void EmitElementwiseKernel(string name, IEnumerable<Node> nodes, long[] shape, string? symbol,
        List<TensorRef> inputs, List<(TensorRef Ref, string Expr)> outputs, List<string> stmts)
    {
        int rank = shape.Length;
        var P = new long[rank];
        long acc = 1;
        for (int k = rank - 1; k >= 0; k--) { P[k] = acc; acc *= shape[k]; }
        long inner = rank == 0 ? 1 : P[0];

        var sb = new StringBuilder();
        sb.AppendLine("#include \"math_ext.slang\"");
        sb.AppendLine("struct Params {");
        for (int i = 0; i < inputs.Count; i++) sb.AppendLine($"  {PtrType(inputs[i].Type)}* in{i};");
        for (int i = 0; i < outputs.Count; i++) sb.AppendLine($"  float* out{i};");
        sb.AppendLine("  uint n;");
        sb.AppendLine("};");
        sb.AppendLine("[shader(\"compute\")] [numthreads(256, 1, 1)]");
        sb.AppendLine("void main(uint3 dtid : SV_DispatchThreadID, uniform Params p) {");
        sb.AppendLine("  uint idx = dtid.x; if (idx >= p.n) return;");
        for (int k = 0; k < rank; k++)
            sb.AppendLine(k == 0 ? $"  uint d0 = idx / {P[0]}u;" : $"  uint d{k} = (idx / {P[k]}u) % {shape[k]}u;");
        for (int i = 0; i < inputs.Count; i++)
            sb.AppendLine($"  float a{i} = {Load(inputs[i], i)};");
        foreach (var s in stmts) sb.AppendLine("  " + s);
        for (int i = 0; i < outputs.Count; i++)
            sb.AppendLine($"  p.out{i}[{Offset(outputs[i].Ref)}] = {outputs[i].Expr};");
        sb.AppendLine("}");

        var kernel = _compiler.Compile("elementwise.slang", sb.ToString(), "main", _opt.Target, Macros(), _opt.Cache);
        long total = shape.Aggregate(1L, (a, b) => a * b);
        KernelArg count = symbol is null ? new UIntArg((uint)total) : new SymbolArg(symbol, inner);
        Emit(name, nodes,
            [Always(kernel, b => new Grid(Grid.Div((symbol is null ? shape[0] : b[symbol]) * inner, 256)))],
            [.. inputs.Select(r => (KernelArg)new PointerArg(r.Buffer, r.ByteOffset)), .. outputs.Select(o => (KernelArg)new PointerArg(o.Ref.Buffer, o.Ref.ByteOffset)), count],
            inputs, outputs.Select(o => o.Ref));

        string Offset(TensorRef r)
        {
            var terms = new List<string>();
            for (int k = 0; k < rank; k++)
            {
                int rk = r.Rank - rank + k;
                if (rk < 0) continue;
                long st = r.Shape[rk] == 1 && shape[k] != 1 ? 0 : r.Strides[rk];
                if (st == 0 || (shape[k] == 1)) continue;
                terms.Add(st == 1 ? $"d{k}" : $"d{k} * {st}u");
            }
            return terms.Count == 0 ? "0" : string.Join(" + ", terms);
        }

        string Load(TensorRef r, int i) => r.Type == DataType.F32 ? $"p.in{i}[{Offset(r)}]"
            : r.Type == DataType.F16 ? $"h2f(p.in{i}[{Offset(r)}])"
            : r.Type == DataType.BF16 ? $"bf2f(p.in{i}[{Offset(r)}])"
            : r.Type == DataType.I32 || r.Type == DataType.I64 || r.Type == DataType.U32 ? $"float(p.in{i}[{Offset(r)}])"
            : throw new NotSupportedException($"elementwise input type {r.Type}");

        static string PtrType(DataType t) => t == DataType.F32 ? "float" : t == DataType.I32 ? "int" : t == DataType.I64 ? "int64_t" : t == DataType.U32 ? "uint" : "uint16_t";
    }
}
