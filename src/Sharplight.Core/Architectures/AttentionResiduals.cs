using Sharplight.Graph;
using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Block attention residuals (Kimi K3): instead of one running residual sum, the residual stream is split into blocks of
/// <c>attention_residuals.block_size</c> layers. The input of a layer at a block start is banked as a checkpoint and the running sum
/// restarts from that layer's attention output; every sub-layer (and the output head) reads a softmax-weighted mix of the banked
/// checkpoints and the current partial sum, scored per token by a learned vector. Pure graph code over standard ops, per token, so it
/// works unchanged with continuous batching.
/// </summary>
/// <remarks>
/// Mix(current) with checkpoints c_1..c_N and v_{N+1} = current: s_i = Σ_d rms_norm(v_i)_d · w_d (unweighted RMS norm, epsilon
/// <c>norm_eps</c>), p = softmax(s), result Σ_i p_i · v_i (the raw values, not the normalized ones). Score weights w [H] (f32): per layer
/// <c>layers.{i}.attention_res.score.weight</c> (before the token mixer) and <c>layers.{i}.mlp_res.score.weight</c> (before the feed-forward
/// block), and <c>output_res.score.weight</c> before the output norm (each is the product of the reference's norm weight and its 1-row
/// projection). A layer whose index is a multiple of the block size banks its raw input before its mixer runs; layer 0 reads its
/// input unmixed (nothing is banked yet).
/// One instance tracks the checkpoints of one graph under construction; <see cref="Reset"/> starts a new graph.
/// Variant (Inkling 2): with <paramref name="normalizedValues"/> the result mixes the normalized rows instead of the raw ones, and every score is
/// multiplied by <paramref name="scoreScale"/> (H^(-1/2) there: the mix is a one-query attention over the normalized rows); the caller decides
/// when it banks (<see cref="Push"/>) and mixes (<see cref="Mix"/>), and may save and restore the bank (<see cref="Save"/>, <see cref="Restore"/>)
/// or open a bank over the checkpoints added since a saved one (<see cref="Since"/>).
/// </remarks>
internal sealed class AttentionResiduals(int blockSize, float eps, bool normalizedValues = false, float scoreScale = 1f)
{
    private Value? _blocks;    // [T, N, H] banked checkpoints
    private Value? _normed;    // [T, N, H] their RMS-normalized rows (scores, and the values with normalizedValues)

    /// <summary>The checkpoints banked so far (null while none), with their normalized rows; see <see cref="Save"/>.</summary>
    /// <param name="Blocks">The banked rows [T, N, H].</param>
    /// <param name="Normed">Their RMS-normalized rows [T, N, H].</param>
    internal readonly record struct Bank(Value? Blocks, Value? Normed)
    {
        /// <summary>The number of banked checkpoints.</summary>
        public int Count => Blocks is null ? 0 : (int)Blocks.Shape[1].Fixed;
    }

    /// <summary>The block size in layers.</summary>
    public int BlockSize => blockSize;

    /// <summary>Reads <c>attention_residuals.block_size</c>; null when the model has no attention residuals (0 or missing).</summary>
    public static AttentionResiduals? Read(ModelSource src, float eps)
    {
        int size = src.Arch("attention_residuals.block_size", 0);
        return size > 0 ? new AttentionResiduals(size, eps) : null;
    }

    /// <summary>Forgets the checkpoints (start of a new graph).</summary>
    public void Reset()
    {
        _blocks = null;
        _normed = null;
    }

    /// <summary>The current bank.</summary>
    public Bank Save() => new(_blocks, _normed);

    /// <summary>Makes <paramref name="bank"/> (from <see cref="Save"/>) the current bank.</summary>
    public void Restore(Bank bank) => (_blocks, _normed) = (bank.Blocks, bank.Normed);

    /// <summary>
    /// A new instance (same settings) whose bank holds the current checkpoints after the first <paramref name="skip"/> ones; nothing when none
    /// remain.
    /// </summary>
    public AttentionResiduals Since(GraphBuilder b, int skip)
    {
        var tail = new AttentionResiduals(blockSize, eps, normalizedValues, scoreScale);
        int count = Save().Count;
        if (_blocks is not null && _normed is not null && count > skip)
        {
            tail._blocks = skip == 0 ? _blocks : b.Slice(_blocks, 1, skip, count - skip);
            tail._normed = skip == 0 ? _normed : b.Slice(_normed, 1, skip, count - skip);
        }

        return tail;
    }

    /// <summary>Whether layer <paramref name="layer"/> starts a block (banks its input and restarts the running sum).</summary>
    public bool StartsBlock(int layer) => layer % blockSize == 0;

    /// <summary>Banks the rows <paramref name="x"/> [T, H] as a checkpoint.</summary>
    public void Push(GraphBuilder b, Value x, int hidden)
    {
        var row = b.Reshape(x, 0, 1, hidden);
        var normed = b.RmsNorm(row, null, eps);
        _blocks = _blocks is null ? row : b.Concat(1, _blocks, row);
        _normed = _normed is null ? normed : b.Concat(1, _normed, normed);
    }

    /// <summary>
    /// The mix of the checkpoints and <paramref name="current"/> [T, H] scored by the weight <paramref name="scoreWeight"/>; the current rows
    /// unchanged while nothing is banked.
    /// </summary>
    public Value Mix(GraphBuilder b, Value current, string scoreWeight, int hidden, string name)
        => Mix(b, current, b.Weight(scoreWeight), hidden, name);

    /// <summary>
    /// The mix of the checkpoints and <paramref name="current"/> [T, H] scored by the vector <paramref name="w"/> [H] (a weight or a row of
    /// one); the current rows unchanged while nothing is banked (with normalized values: their normalized rows).
    /// </summary>
    public Value Mix(GraphBuilder b, Value current, Value w, int hidden, string name)
    {
        if (_blocks is null || _normed is null)
        {
            return normalizedValues ? b.RmsNorm(current, null, eps) : current;
        }

        var currentNormed = b.RmsNorm(current, null, eps);

        int banked = (int)_blocks.Shape[1].Fixed;
        var blockScores = b.Reshape(b.Emit1("ReduceSum", [b.Mul(_normed, w)]), 0, banked);                                    // [T, N]
        var currentScore = b.Emit1("ReduceSum", [b.Mul(currentNormed, w)]);                                                    // [T, 1]
        var scores = b.Concat(1, blockScores, currentScore);
        var probs = b.Emit1("Softmax", [scoreScale == 1f ? scores : b.Scale(scores, scoreScale)], null, $"{name}_probs");     // [T, N + 1]
        var values = normalizedValues ? _normed : _blocks;
        var weighted = b.Mul(values, b.Reshape(b.Slice(probs, 1, 0, banked), 0, banked, 1));                                   // [T, N, H]
        var fromBlocks = b.Reshape(b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]), 0, hidden);
        return b.Add(fromBlocks, b.Mul(normalizedValues ? currentNormed : current, b.Slice(probs, 1, banked, 1)), name);
    }
}
