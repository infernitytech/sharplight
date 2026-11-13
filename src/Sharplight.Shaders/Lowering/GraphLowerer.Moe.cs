using System.Globalization;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Lowering of the expert-indexed matrix product (MatMulId, see kernels/moe.slang).</summary>
public sealed partial class GraphLowerer
{
    private const int MoeMaxExperts = 1024;
    private const int MoeMinTile = 32;

    private static OpSupport SupportsMoe(Node node)
    {
        var x = node.Inputs[0];
        var w = node.Inputs[1];
        var ids = node.Inputs[2];
        if (x.Type != DataType.F32)
        {
            return OpSupport.No($"MatMulId activations must be f32, got {x.Type}");
        }

        if (ids.Type != DataType.I32)
        {
            return OpSupport.No("MatMulId expert indices must be i32");
        }

        if (w.Kind != ValueKind.Weight)
        {
            return OpSupport.No("MatMulId weight must be a model parameter");
        }

        if (WeightFormat(w.Type) is null)
        {
            return OpSupport.No($"MatMulId weight type {w.Type} not supported");
        }

        if (w.Shape.Rank != 3 || w.Shape.Dims.Any(d => d.IsSymbolic))
        {
            return OpSupport.No("MatMulId weight shape must be static [E, N, K]");
        }

        if (w.Shape[0].Fixed > MoeMaxExperts)
        {
            return OpSupport.No($"MatMulId with more than {MoeMaxExperts} experts");
        }

        if (w.Shape[2].Fixed % 8 != 0)
        {
            return OpSupport.No($"MatMulId K={w.Shape[2].Fixed} is not a multiple of 8");
        }

        if (ids.Shape[1].IsSymbolic)
        {
            return OpSupport.No("MatMulId slot count must be static");
        }

        return OpSupport.Yes;
    }

    // Routing (expert tiles + sorted items) is shared by the MatMulIds of one expert block (same ids, same E, same tiling: the
    // tensor-core path depends on K, so the projections of one block may tile differently).
    private BufferDecl? _moeScratch;
    private (Value Ids, long E, long MaxTiles, bool Coop)? _moeRouted;

    private void LowerMatMulId(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var w = Ref(n.Inputs[1]);
        var ids = Ref(n.Inputs[2]);
        long E = w.Shape[0];
        long N = w.Shape[1];
        long K = w.Shape[2];
        long S = ids.Shape[1];
        bool perSlot = x.Rank == 3;
        if (!ids.IsDense || ids.Type != DataType.I32)
        {
            ids = DenseTemp(ids, "matmul-id ids");
        }

        x = PrepareMatMulInput(x, "matmul-id", out long xs, out long rowsPerLead);
        var y = Produce(n.Output);
        int fmt = WeightFormat(w.Type)!.Value;
        long rowBytes = w.Type.ByteSize(K);
        var m = Macros(("WFMT", fmt), ("K", K), ("N", N), ("E", E), ("S", S), ("PER_SLOT", perSlot), ("X_STRIDE", xs),
            ("ROW_BYTES", rowBytes + "ull"), ("EXPERT_BYTES", rowBytes * N + "ull"));
        var tids = ids;
        var tx = x;
        long maxItems = ids.Shape[0] * S;
        Func<Bindings, long> items = b => Rows(b, tids) * S;
        bool coop = !DiagnosticSwitches.MoeNoCoop && _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16) && _opt.SubgroupSize == 32 && K % 32 == 0
                    && (w.Type != DataType.F32 || _opt.Compile.AllowReducedPrecision);
        // Items up to which the per-item matvec is used (tensor-core GEMMs win from a few rows, the scalar tiled GEMM much
        // later, the int-dot matvec only from about 16 items per expert).
        bool intDot = UseIntDotMatvec(w.Type, K);
        long mv = intDot ? DiagnosticSwitches.MoeIntDotMaxItems
            : coop ? DiagnosticSwitches.MoeMatvecMaxItems : Math.Max(DiagnosticSwitches.MoeMatvecMaxItems, DiagnosticSwitches.MoeScalarRows * E);
        string label = $"matmul-id {n.Output.Name} [{E}x{K}→{N}] {w.Type}";

        // Decode / few rows: one matvec per (token, slot), with int8 dot products when the format allows.
        if (intDot)
        {
            EmitIntDotExpertMatvec(label, n, x, xs, rowsPerLead, w, ids, y, m, mv, S);
        }

        Emit(label + " matvec", [n],
            [new KernelVariant(b => items(b) <= mv && !intDot, Lib("moe.slang", "moe_matvec", m), b => new Grid(Grid.Div(N, 4), (uint)Math.Max(1, items(b))))],
            [Pointer(x), Pointer(w), Pointer(ids),
             Pointer(y), TokenArg(ids)],
            [x, w, ids], [y]);

        // Many rows: group by expert, then a GEMM per expert tile.
        long maxTiles = Math.Min(maxItems, maxItems / MoeMinTile + E);
        long scratchWords = 1 + 3 * maxTiles + maxItems;
        _moeScratch ??= NewBuffer(BufferKind.Activations, "moe-routing", 0);
        _moeScratch.Bytes = Math.Max(_moeScratch.Bytes, Math.Max(16, scratchWords * 4));
        var scratch = new TensorRef(_moeScratch, 0, DataType.U32, [scratchWords], [1], null);

        // Tile rows per expert: small tiles when experts see few rows on average (less wasted MMA work). The grouped gemm_cm2
        // also has 128-row tiles for experts that see many rows.
        bool gemm2 = coop && DiagnosticSwitches.MoeGemm2;
        int[] coopTiles = gemm2 ? [32, 64, 128] : [32, 64];
        Func<Bindings, int> tile = b => !coop ? 64
            : DiagnosticSwitches.MoeTile != 0 ? DiagnosticSwitches.MoeTile
            : items(b) <= 24 * E ? 32
            : !gemm2 || items(b) <= 160 * E ? 64 : 128;
        Func<Bindings, uint> tiles = b => (uint)Math.Max(1, Math.Min(items(b), items(b) / tile(b) + E));
        var scratchArg = new PointerArg(_moeScratch, 0);

        if (_moeRouted is not { } r || r.Ids != n.Inputs[2] || r.E != E || r.MaxTiles != maxTiles || r.Coop != coop)
        {
            int[] tiling = coop ? coopTiles : [64];
            Emit($"moe-route {n.Inputs[2].Name} [{E} experts]", [n],
                [.. tiling.Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t,
                    Lib("moe.slang", "moe_route", new Dictionary<string, string>(Macros(("E", E), ("S", S), ("TILE_M", t), ("MAXTILES", maxTiles)))),
                    _ => new Grid(1)))],
                [Pointer(ids), scratchArg, TokenArg(ids)],
                [ids], [scratch]);
            _moeRouted = (n.Inputs[2], E, maxTiles, coop);
        }

        var gm = new Dictionary<string, string>(m) { ["MAXTILES"] = maxTiles.ToString(CultureInfo.InvariantCulture) };
        var gargs = new KernelArg[] { Pointer(x), Pointer(w), scratchArg, Pointer(y), TokenArg(ids) };
        if (gemm2)
        {
            // The dense tensor-core GEMM in its expert-grouped form: one routing tile per workgroup row.
            var groupedMacros = new Dictionary<string, string>(gm) { ["MOE"] = "1", ["Y_STRIDE"] = N.ToString(CultureInfo.InvariantCulture) };
            Emit(label + " gemm-cm2", [n],
                [.. coopTiles.Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t,
                    Lib("matmul.slang", "gemm_cm2", Gemm2Macros(groupedMacros, t, large: false)),
                    b => new Grid(Grid.Div(N, Gemm2Columns(t)), tiles(b))))],
                gargs, [x, w, scratch], [y]);
        }
        else if (coop)
        {
            Emit(label + " gemm-cm", [n],
                [.. new[] { 32, 64 }.Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t,
                    Lib("moe.slang", "moe_gemm_cm", new Dictionary<string, string>(gm)
                    {
                        ["WARPS_M"] = "2", ["WARPS_N"] = "2", ["FRAG_M"] = (t / 32).ToString(CultureInfo.InvariantCulture), ["FRAG_N"] = "4",
                    }),
                    b => new Grid(Grid.Div(N, 128), tiles(b))))],
                gargs, [x, w, scratch], [y]);
        }
        else
        {
            Emit(label + " gemm", [n],
                [new KernelVariant(b => items(b) > mv, Lib("moe.slang", "moe_gemm", gm), b => new Grid(Grid.Div(N, 64), tiles(b)))],
                gargs, [x, w, scratch], [y]);
        }
    }
}
