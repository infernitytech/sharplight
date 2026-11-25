using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Partial-sum buffers of the stream-K launches, used in rotation (see <see cref="StreamKBuffers"/>).</summary>
    private readonly BufferDecl?[] _streamKScratch = new BufferDecl?[3];

    /// <summary>Flags of the stream-K launches, one per partial-sum buffer.</summary>
    private readonly BufferDecl?[] _streamKFlags = new BufferDecl?[3];

    private int _streamKNext;

    /// <summary>
    /// The partial sums (grown to <paramref name="bytes"/>) and the flags (<paramref name="blocks"/> u32: zero when created, and every
    /// launch leaves them zero) of the next stream-K launch. Apart from the split-K scratch, so that the buffers of launches that never
    /// run in the same steps (split-K at decode sizes) do not change; consecutive launches get different buffers (three in rotation),
    /// so sibling matmuls do not order each other through them.
    /// </summary>
    private (BufferDecl Scratch, TensorRef Flags) StreamKBuffers(long bytes, long blocks)
    {
        int slot = _streamKNext++ % _streamKScratch.Length;
        var scratch = _streamKScratch[slot] ??= NewBuffer(BufferKind.Activations, "streamk" + slot, 0);
        scratch.Bytes = Math.Max(scratch.Bytes, bytes);
        long flagBytes = Math.Max(16, blocks * 4);
        if (_streamKFlags[slot] is not { } flags || flags.Bytes < flagBytes)
        {
            flags = _streamKFlags[slot] = NewBuffer(BufferKind.Weight, "const:streamk-flags" + slot, flagBytes, init: new byte[flagBytes]);
        }

        return (scratch, new TensorRef(flags, 0, DataType.U32, [flagBytes / 4], [1], null));
    }

    /// <summary>
    /// Estimated stream-K costs of the FP8 tensor-copy GEMM (gemm_fp8_tma.cu <c>F8N_SK</c>): a K step of the shared range costs 1.12 of a
    /// tile-by-tile step and a block's fixup 4 steps. Fitted on the RTX PRO 4500 Blackwell (Qwen3-8B-FP8 projections at 256 and 266 rows,
    /// benchmarks.md section 104: shared-range time = 1.11-1.14 x steps + 4).
    /// </summary>
    private static readonly (double Step, double Fixup) s_fp8StreamKCost = (1.12, 4);

    /// <summary>
    /// Estimated stream-K costs of the hand-written f16 GEMM (<c>H_SK</c>): a K step as a tile-by-tile one, a block's fixup 8 steps.
    /// Llama-3.1-8B, Qwen3-8B and gpt-oss-20b prompt steps of 128-2,048 rows (benchmarks.md section 104): the gains at 256-2,048 rows
    /// hold from 4 to 21 steps; above 8 the q / k / v GEMMs at 256 rows lose stream-K, below it a 2,880-output projection at 266 rows is
    /// 9 % slower stream-K.
    /// </summary>
    private static readonly (double Step, double Fixup) s_f16StreamKCost = (1.0, 8);

    /// <summary>Fewest K steps of a stream-K block's range: shorter ranges are mostly fixup.</summary>
    private const long StreamKMinSteps = 8;

    /// <summary>
    /// Tiles a stream-K launch of <paramref name="tiles"/> output tiles on <paramref name="blocks"/> blocks walks whole, in full waves
    /// (every full wave but the last; none below two waves); the rest share their K steps evenly (the kernels' split, e.g.
    /// gemm_fp8_tma.cu <c>F8N_SK</c>).
    /// </summary>
    private static long StreamKWholeTiles(long tiles, long blocks) => tiles >= 2 * blocks ? (tiles / blocks - 1) * blocks : 0;

    /// <summary>
    /// Most token tiles of a stream-K launch: the kernels walk the shared tiles weight tile by weight tile in lockstep only when the
    /// token tiles fit one group of their rasterized order (16).
    /// </summary>
    private const long StreamKMaxTokenTiles = 16;

    /// <summary>
    /// Blocks of a stream-K launch with <paramref name="tokenTiles"/> token tiles: the most SMs that are a multiple of the token tiles,
    /// so that the blocks reading one weight tile for different token tiles read each of its K steps at the same time (the kernels'
    /// order of the shared tiles); 0 when the launch does not run stream-K. Not for a single token tile: each weight tile is then
    /// read by one block only, the launch streams the weights at the memory's pace, and more blocks splitting K only add their
    /// fixups (Qwen3-8B-FP8 at 128 rows: every projection 2-15 % slower stream-K).
    /// </summary>
    private long StreamKBlocks(long tokenTiles)
    {
        long sms = Math.Max(1, _opt.ComputeUnits);
        return tokenTiles is < 2 or > StreamKMaxTokenTiles || tokenTiles > sms ? 0 : sms - sms % tokenTiles;
    }

    /// <summary>
    /// The estimated time, in K steps of one block, of a launch of <paramref name="tiles"/> output tiles (<paramref name="tokenTiles"/>
    /// token tiles of them) of <paramref name="kSteps"/> K steps each on persistent blocks (at most one per SM): walked tile by tile
    /// (<paramref name="streamK"/> false: the busiest SM's whole tiles) or stream-K on <see cref="StreamKBlocks"/> blocks (the whole tiles
    /// of the full waves, then an even share of the rest's K steps at the kernel's <paramref name="cost"/> plus its fixup). Infinity for a
    /// stream-K launch that does not apply or whose blocks would get fewer than <see cref="StreamKMinSteps"/> K steps of the shared range.
    /// </summary>
    private double TileWalkSteps(long tiles, long tokenTiles, long kSteps, bool streamK, (double Step, double Fixup) cost)
    {
        long sms = Math.Max(1, _opt.ComputeUnits);
        if (!streamK)
        {
            return (double)Grid.Div(tiles, sms) * kSteps;
        }

        long blocks = StreamKBlocks(tokenTiles);
        if (blocks == 0)
        {
            return double.PositiveInfinity;
        }

        long whole = StreamKWholeTiles(tiles, blocks);
        long shared = (tiles - whole) * kSteps;
        if (shared < blocks * StreamKMinSteps)
        {
            return double.PositiveInfinity;
        }

        return whole / blocks * kSteps + Grid.Div(shared, blocks) * cost.Step + (DiagnosticSwitches.StreamKCost ?? cost.Fixup);
    }

    /// <summary>
    /// True when a launch of <paramref name="tiles"/> output tiles (<paramref name="tokenTiles"/> token tiles) of <paramref name="kSteps"/>
    /// K steps runs stream-K: its tiles do not fill whole waves of the SMs and the even split of the last waves' K steps is estimated
    /// faster than walking them tile by tile, by at least <see cref="StreamKMinGain"/> (keyed on tile and wave counts and K only).
    /// <c>SHARPLIGHT_STREAM_K=0</c> turns it off.
    /// </summary>
    private bool UseStreamK(long tiles, long tokenTiles, long kSteps, (double Step, double Fixup) cost)
        => StreamKEnabled
           && TileWalkSteps(tiles, tokenTiles, kSteps, true, cost) < TileWalkSteps(tiles, tokenTiles, kSteps, false, cost) * (1 - StreamKMinGain);

    /// <summary>False with <c>SHARPLIGHT_STREAM_K=0</c>: no launch runs stream-K.</summary>
    private static bool StreamKEnabled => DiagnosticSwitches.StreamK ?? true;

    /// <summary>
    /// Least estimated gain for which a launch runs stream-K: within it the estimate does not tell the walks apart (and stream-K changes
    /// the summation order for nothing).
    /// </summary>
    private const double StreamKMinGain = 0.05;

    /// <summary>
    /// Emits the stream-K launch of a hand-written GEMM (<paramref name="variants"/>, its stream-K variants) with the <paramref name="args"/>
    /// of its tile-by-tile launch (pointers, the token count last) and the partial sums (<paramref name="slotBytes"/> per SM's block) and
    /// flags, after them. Nothing when there are no variants.
    /// </summary>
    private void EmitStreamK(string label, IEnumerable<Node> nodes, KernelVariant[] variants, KernelArg[] args, TensorRef[] reads, TensorRef[] writes,
        long slotBytes)
    {
        if (variants.Length == 0)
        {
            return;
        }

        long blocks = Math.Max(1, _opt.ComputeUnits);
        var (scratch, flags) = StreamKBuffers(blocks * slotBytes, blocks);
        var partial = new TensorRef(scratch, 0, DataType.F32, [blocks * slotBytes / 4], [1], null);
        Emit(label + " stream-k", nodes, variants, [.. args[..^1], new PointerArg(scratch, 0), Pointer(flags), args[^1]], reads, [.. writes, partial, flags]);
    }

    /// <summary>The estimated time of <see cref="TileWalkSteps"/> with the walk <see cref="UseStreamK"/> picks.</summary>
    private double BestTileWalkSteps(long tiles, long tokenTiles, long kSteps, (double Step, double Fixup) cost)
        => TileWalkSteps(tiles, tokenTiles, kSteps, UseStreamK(tiles, tokenTiles, kSteps, cost), cost);
}
