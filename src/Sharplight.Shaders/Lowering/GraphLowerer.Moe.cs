using System.Globalization;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Lowering of the expert-indexed matrix product (MatMulId, see kernels/moe.slang).</summary>
public sealed partial class GraphLowerer
{
    const int MoeMatvecMaxItems = 16;   // (token, slot) items up to which a per-item matvec beats grouping by expert
    const int MoeMaxExperts = 1024;
    const int MoeMinTile = 32;

    static readonly int s_moeMvMax = int.Parse(Environment.GetEnvironmentVariable("SHARPLIGHT_MOE_MV_MAX") ?? MoeMatvecMaxItems.ToString(CultureInfo.InvariantCulture), CultureInfo.InvariantCulture);
    static readonly int s_moeTile = int.Parse(Environment.GetEnvironmentVariable("SHARPLIGHT_MOE_TILE") ?? "0", CultureInfo.InvariantCulture);
    static readonly bool s_moeNoCoop = Environment.GetEnvironmentVariable("SHARPLIGHT_MOE_NO_COOP") == "1";

    static OpSupport SupportsMoe(Node node)
    {
        var x = node.Inputs[0]; var w = node.Inputs[1]; var ids = node.Inputs[2];
        if (x.Type != DataType.F32) return OpSupport.No($"MatMulId activations must be f32, got {x.Type}");
        if (ids.Type != DataType.I32) return OpSupport.No("MatMulId expert indices must be i32");
        if (w.Kind != ValueKind.Weight) return OpSupport.No("MatMulId weight must be a model parameter");
        if (WeightFormat(w.Type) is null) return OpSupport.No($"MatMulId weight type {w.Type} not supported");
        if (w.Shape.Rank != 3 || w.Shape.Dims.Any(d => d.IsSymbolic)) return OpSupport.No("MatMulId weight shape must be static [E, N, K]");
        if (w.Shape[0].Fixed > MoeMaxExperts) return OpSupport.No($"MatMulId with more than {MoeMaxExperts} experts");
        if (w.Shape[2].Fixed % 8 != 0) return OpSupport.No($"MatMulId K={w.Shape[2].Fixed} is not a multiple of 8");
        if (ids.Shape[1].IsSymbolic) return OpSupport.No("MatMulId slot count must be static");
        return OpSupport.Yes;
    }

    // Routing (expert tiles + sorted items) is shared by the MatMulIds of one expert block (same ids, same E).
    BufferDecl? _moeScratch;
    (Value Ids, long E, long MaxTiles)? _moeRouted;

    void LowerMatMulId(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var w = Ref(n.Inputs[1]);
        var ids = Ref(n.Inputs[2]);
        long E = w.Shape[0], N = w.Shape[1], K = w.Shape[2], S = ids.Shape[1];
        bool perSlot = x.Rank == 3;
        if (!ids.IsDense || ids.Type != DataType.I32) ids = DenseTemp(ids, "matmul-id ids");
        if (!TryRows(x, out long xs, out long mid) || xs % 4 != 0 || x.ByteOffset % 16 != 0 || x.Type != DataType.F32)
        {
            x = DenseTemp(x, "matmul-id");
            TryRows(x, out xs, out mid);
        }
        var y = Produce(n.Output);
        int fmt = WeightFormat(w.Type)!.Value;
        long rowBytes = w.Type.ByteSize(K);
        var m = Macros(("WFMT", fmt), ("K", K), ("N", N), ("E", E), ("S", S), ("PER_SLOT", perSlot), ("X_STRIDE", xs),
            ("ROW_BYTES", rowBytes + "ull"), ("EXPERT_BYTES", rowBytes * N + "ull"));
        var tids = ids; var tx = x;
        long maxItems = ids.Shape[0] * S;
        Func<Bindings, long> items = b => Rows(b, tids) * S;
        long mv = s_moeMvMax;
        string label = $"matmul-id {n.Output.Name} [{E}x{K}→{N}] {w.Type}";

        // Decode / few rows: one matvec per (token, slot).
        Emit(label + " matvec", [n],
            [new KernelVariant(b => items(b) <= mv, Lib("moe.slang", "moe_matvec", m), b => new Grid(Grid.Div(N, 4), (uint)Math.Max(1, items(b))))],
            [new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(w.Buffer, w.ByteOffset), new PointerArg(ids.Buffer, ids.ByteOffset),
             new PointerArg(y.Buffer, y.ByteOffset), TokenArg(ids)],
            [x, w, ids], [y]);

        // Many rows: group by expert, then a GEMM per expert tile.
        bool coop = !s_moeNoCoop && _opt.Has(DeviceFeatures.MatrixF16) && _opt.Has(DeviceFeatures.Float16) && _opt.SubgroupSize == 32 && K % 32 == 0
                    && (w.Type != DataType.F32 || _opt.Compile.AllowReducedPrecision);
        long maxTiles = Math.Min(maxItems, maxItems / MoeMinTile + E);
        long scratchWords = 1 + 3 * maxTiles + maxItems;
        _moeScratch ??= NewBuffer(BufferKind.Activations, "moe-routing", 0);
        _moeScratch.Bytes = Math.Max(_moeScratch.Bytes, Math.Max(16, scratchWords * 4));
        var scratch = new TensorRef(_moeScratch, 0, DataType.U32, [scratchWords], [1], null);

        // Tile rows per expert: small tiles when experts see few rows on average (less wasted MMA work).
        Func<Bindings, int> tile = b => !coop ? 64 : s_moeTile != 0 ? s_moeTile : items(b) <= 24 * E ? 32 : 64;
        Func<Bindings, uint> tiles = b => (uint)Math.Max(1, Math.Min(items(b), items(b) / tile(b) + E));
        var scratchArg = new PointerArg(_moeScratch, 0);

        if (_moeRouted is not { } r || r.Ids != n.Inputs[2] || r.E != E || r.MaxTiles != maxTiles)
        {
            var tiling = coop ? new[] { 32, 64 } : new[] { 64 };
            Emit($"moe-route {n.Inputs[2].Name} [{E} experts]", [n],
                [.. tiling.Select(t => new KernelVariant(b => items(b) > mv && tile(b) == t,
                    Lib("moe.slang", "moe_route", new Dictionary<string, string>(Macros(("E", E), ("S", S), ("TILE_M", t), ("MAXTILES", maxTiles)))),
                    _ => new Grid(1)))],
                [new PointerArg(ids.Buffer, ids.ByteOffset), scratchArg, TokenArg(ids)],
                [ids], [scratch]);
            _moeRouted = (n.Inputs[2], E, maxTiles);
        }

        var gm = new Dictionary<string, string>(m) { ["MAXTILES"] = maxTiles.ToString(CultureInfo.InvariantCulture) };
        var gargs = new KernelArg[] { new PointerArg(x.Buffer, x.ByteOffset), new PointerArg(w.Buffer, w.ByteOffset), scratchArg, new PointerArg(y.Buffer, y.ByteOffset), TokenArg(ids) };
        if (coop)
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
