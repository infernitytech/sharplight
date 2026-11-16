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
/// </remarks>
internal sealed class AttentionResiduals(int blockSize, float eps)
{
    private Value? _blocks;    // [T, N, H] banked checkpoints
    private Value? _normed;    // [T, N, H] their RMS-normalized rows (scores only)

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
    {
        if (_blocks is null || _normed is null)
        {
            return current;
        }

        var w = b.Weight(scoreWeight);
        int banked = (int)_blocks.Shape[1].Fixed;
        var blockScores = b.Reshape(b.Emit1("ReduceSum", [b.Mul(_normed, w)]), 0, banked);                                    // [T, N]
        var currentScore = b.Emit1("ReduceSum", [b.Mul(b.RmsNorm(current, null, eps), w)]);                                    // [T, 1]
        var probs = b.Emit1("Softmax", [b.Concat(1, blockScores, currentScore)], null, $"{name}_probs");                       // [T, N + 1]
        var weighted = b.Mul(_blocks, b.Reshape(b.Slice(probs, 1, 0, banked), 0, banked, 1));                                  // [T, N, H]
        var fromBlocks = b.Reshape(b.Emit1("ReduceSum", [b.Permute(weighted, 0, 2, 1)]), 0, hidden);
        return b.Add(fromBlocks, b.Mul(current, b.Slice(probs, 1, banked, 1)), name);
    }
}
