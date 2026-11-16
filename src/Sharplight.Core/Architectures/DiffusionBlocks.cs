using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Building blocks of convolutional diffusion networks (UNets and VAEs of latent diffusion, and the video / image models that reuse
/// them), on channels-first activations [N, C, H, W] with a symbolic batch N: group norms, 3×3 / 1×1 convolutions, residual blocks
/// with an optional time-embedding injection, nearest upsampling, strided downsampling and the conversion between feature maps and
/// the token rows [H·W·N, C] that attention layers work on.
/// </summary>
internal static class DiffusionBlocks
{
    /// <summary>Group normalization with the scale and bias weights <c>{prefix}.weight</c> / <c>{prefix}.bias</c>.</summary>
    public static Value GroupNorm(GraphBuilder b, Value x, string prefix, int groups, float eps)
        => b.Emit1("GroupNorm", [x, b.Weight(prefix + ".weight"), b.Weight(prefix + ".bias")], Attributes.From(("groups", (long)groups), ("eps", eps)));

    /// <summary>Layer normalization over the last axis with the weights <c>{prefix}.weight</c> / <c>{prefix}.bias</c>.</summary>
    public static Value LayerNorm(GraphBuilder b, Value x, string prefix, float eps)
        => b.Emit1("LayerNorm", [x, b.Weight(prefix + ".weight"), b.Weight(prefix + ".bias")], Attributes.From(("eps", eps)));

    /// <summary>
    /// 2-D convolution with <c>{prefix}.weight</c> [O, C, K, K] and the optional <c>{prefix}.bias</c>, "same" padding for odd kernels
    /// unless <paramref name="pads"/> (top, left, bottom, right) is given.
    /// </summary>
    public static Value Conv(GraphBuilder b, Value x, string prefix, long stride = 1, long[]? pads = null)
    {
        var weight = b.Weight(prefix + ".weight");
        long half = weight.Shape[2].Fixed / 2;
        var attributes = Attributes.From(("strides", new[] { stride, stride }), ("pads", pads ?? [half, half, half, half]));
        var bias = b.TryWeight(prefix + ".bias");
        return b.Emit1("Conv", bias is null ? [x, weight] : [x, weight, bias], attributes);
    }

    /// <summary>
    /// Residual block: GroupNorm, SiLU, 3×3 conv, plus the projected time embedding when given, GroupNorm, SiLU, 3×3 conv, added to
    /// the input (through the 1×1 <c>{prefix}.shortcut</c> conv when the channel count changes).
    /// </summary>
    /// <param name="b">The graph.</param>
    /// <param name="x">Input [N, C, H, W].</param>
    /// <param name="timeEmbedding">SiLU of the time embedding [N, E], or <see langword="null"/> (VAE blocks).</param>
    /// <param name="prefix">Weight prefix (<c>…res.{i}</c>).</param>
    /// <param name="groups">GroupNorm groups.</param>
    /// <param name="eps">GroupNorm epsilon.</param>
    public static Value ResidualBlock(GraphBuilder b, Value x, Value? timeEmbedding, string prefix, int groups, float eps)
    {
        var h = Conv(b, b.Silu(GroupNorm(b, x, prefix + ".norm1", groups, eps)), prefix + ".conv1");
        long channels = h.Shape[1].Fixed;
        if (timeEmbedding is not null)
        {
            var projected = b.Linear(timeEmbedding, prefix + ".time_proj.weight", prefix + ".time_proj.bias");
            h = b.Add(h, b.Reshape(projected, 0, channels, 1, 1));
        }

        h = Conv(b, b.Silu(GroupNorm(b, h, prefix + ".norm2", groups, eps)), prefix + ".conv2");
        var shortcut = b.HasWeight(prefix + ".shortcut.weight") ? Conv(b, x, prefix + ".shortcut") : x;
        return b.Add(shortcut, h);
    }

    /// <summary>Nearest-neighbour 2× upsampling followed by the 3×3 conv <c>{prefix}</c>.</summary>
    public static Value Upsample(GraphBuilder b, Value x, string prefix)
    {
        long[] size = [x.Shape[2].Fixed * 2, x.Shape[3].Fixed * 2];
        var up = b.Emit1("Resize", [x], Attributes.From(("size", size), ("mode", "nearest"), ("coord", "asymmetric"), ("nearest_mode", "floor")));
        return Conv(b, up, prefix);
    }

    /// <summary>
    /// Feature map [N, C, H, W] → token rows [H·W·N, C] (each image's H·W rows consecutive), the layout of the batched attention form.
    /// </summary>
    public static Value ToRows(GraphBuilder b, Value x)
    {
        long channels = x.Shape[1].Fixed;
        long pixels = x.Shape[2].Fixed * x.Shape[3].Fixed;
        var flat = b.Reshape(x, 0, channels, pixels);
        var transposed = b.Contiguous(b.Permute(flat, 0, 2, 1));
        return b.Emit1("Reshape", [transposed], Attributes.From(("shape", new[] { 0, channels }), ("outer_symbol", $"{pixels}*{BatchOf(x)}")));
    }

    /// <summary>Token rows [H·W·N, C] → feature map [N, C, H, W] (the inverse of <see cref="ToRows"/>).</summary>
    public static Value FromRows(GraphBuilder b, Value rows, Dim batch, long height, long width)
    {
        long channels = rows.Shape[1].Fixed;
        var perImage = b.Emit1("Reshape", [rows], Attributes.From(("shape", new[] { 0, height * width, channels }), ("outer_symbol", batch.ToString())));
        var transposed = b.Contiguous(b.Permute(perImage, 0, 2, 1));
        return b.Reshape(transposed, 0, channels, height, width);
    }

    /// <summary>
    /// Multi-head attention of query rows over key / value rows in the batched form: each of the N images' <paramref name="queryRows"/>
    /// rows sees only its own <paramref name="keyRows"/> keys. Projections <c>{prefix}.q/k/v/o</c> (biases where present).
    /// </summary>
    /// <param name="b">The graph.</param>
    /// <param name="x">Query input rows [Q·N, C].</param>
    /// <param name="context">Key / value input rows [K·N, Ck] (<paramref name="x"/> for self-attention).</param>
    /// <param name="prefix">Weight prefix.</param>
    /// <param name="heads">Attention heads.</param>
    /// <param name="queryRows">Query rows per image.</param>
    /// <param name="keyRows">Key rows per image.</param>
    public static Value Attention(GraphBuilder b, Value x, Value context, string prefix, int heads, long queryRows, long keyRows)
    {
        var q = b.Linear(x, prefix + ".q.weight", prefix + ".q.bias");
        var k = b.Linear(context, prefix + ".k.weight", prefix + ".k.bias");
        var v = b.Linear(context, prefix + ".v.weight", prefix + ".v.bias");
        long inner = q.Shape[1].Fixed;
        long headDim = inner / heads;
        var attended = b.Emit1("Attention", [b.Reshape(q, 0, heads, headDim), b.Reshape(k, 0, heads, headDim), b.Reshape(v, 0, heads, headDim)],
            Attributes.From(("scale", 1f / MathF.Sqrt(headDim)), ("causal", false), (BatchedAttention.RowsAttribute, queryRows),
                (BatchedAttention.KeysAttribute, keyRows)));
        return b.Linear(b.Reshape(attended, 0, inner), prefix + ".o.weight", prefix + ".o.bias");
    }

    /// <summary>The batch dimension of a feature map or a batch-major tensor.</summary>
    public static Dim BatchOf(Value x) => x.Shape[0];
}
