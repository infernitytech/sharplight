using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// The DaViT vision tower with Florence-2's multi-modal projector, built into an encoder graph: four stages of convolutional patch embeddings (a pre- or post-norm
/// convolution), each with blocks of a spatial block (depthwise convolution, window attention, depthwise convolution, MLP) and a channel block (depthwise convolution, attention
/// across channel groups, depthwise convolution, MLP). The last stage's map gets learned 2-D position embeddings and the cosine temporal embedding of the first frame, then
/// becomes 1 + H·W rows (the spatial average and the tokens), projected and normalized to the language model's width.
/// </summary>
/// <remarks>
/// Hyper-parameters (<c>encoder_decoder.vision.*</c>, comma-separated per stage): <c>depths</c>, <c>embed_dims</c>, <c>head_counts</c>, <c>group_counts</c>, <c>patch_sizes</c>, <c>strides</c>,
/// <c>paddings</c>, <c>pre_norm</c> (0 / 1), and <c>window</c>, <c>mlp_ratio</c>, <c>image_size</c>, <c>norm_eps</c>, <c>mlp.activation</c>. Weights (<c>vision.</c>): <c>convs.{s}.{conv, norm}</c>,
/// <c>blocks.{s}.{j}.{spatial_block, channel_block}.*</c> (as in transformers), <c>projector.{image_projection (no bias), image_proj_norm, image_position_embed.{row, column}_embeddings}</c>.
/// </remarks>
internal sealed class DaViTGraph
{
    private readonly GraphBuilder _b;
    private readonly ConstantCache _k;
    private readonly int[] _depths;
    private readonly int[] _dims;
    private readonly int[] _heads;
    private readonly int[] _groups;
    private readonly int[] _patches;
    private readonly int[] _strides;
    private readonly int[] _paddings;
    private readonly int[] _preNorm;
    private readonly int _window;
    private readonly float _eps;
    private readonly string _activation;

    public DaViTGraph(ModelSource src, GraphBuilder b, ConstantCache k)
    {
        _b = b;
        _k = k;
        int[] List(string key) => [.. src.Arch<string>("vision." + key).Split(',').Select(int.Parse)];
        _depths = List("depths");
        _dims = List("embed_dims");
        _heads = List("head_counts");
        _groups = List("group_counts");
        _patches = List("patch_sizes");
        _strides = List("strides");
        _paddings = List("paddings");
        _preNorm = List("pre_norm");
        _window = src.Arch<int>("vision.window");
        _eps = src.Arch("vision.norm_eps", 1e-5f);
        _activation = src.Arch("vision.mlp.activation", "gelu");
    }

    /// <summary>The number of rows the tower produces for a square image of <paramref name="imageSize"/> pixels.</summary>
    public int Rows(int imageSize)
    {
        int side = imageSize;
        for (int s = 0; s < _depths.Length; s++)
        {
            side = (side + (2 * _paddings[s]) - _patches[s]) / _strides[s] + 1;
        }

        return (side * side) + 1;
    }

    /// <summary>The projected rows [1 + H·W, hidden] of an image [1, 3, size, size].</summary>
    public Value Build(Value pixels)
    {
        var map = pixels;
        int h = (int)pixels.Shape[2].Fixed;
        int w = (int)pixels.Shape[3].Fixed;
        for (int s = 0; s < _depths.Length; s++)
        {
            using var stage = _b.Group($"vision.stage.{s}");
            int c = _dims[s];
            map = Embed(map, s);
            h = (h + (2 * _paddings[s]) - _patches[s]) / _strides[s] + 1;
            w = (w + (2 * _paddings[s]) - _patches[s]) / _strides[s] + 1;
            for (int j = 0; j < _depths[s]; j++)
            {
                string p = $"vision.blocks.{s}.{j}.";
                map = Spatial(map, p + "spatial_block.", s, c, h, w);
                map = Channel(map, p + "channel_block.", s, c, h, w);
            }
        }

        return Project(map, _dims[^1], h, w);
    }

    // Convolutional embedding of a map; the norm runs over channels before (pre-norm) or after the convolution.
    private Value Embed(Value map, int s)
    {
        string p = $"vision.convs.{s}.";
        int inChannels = (int)map.Shape[1].Fixed;
        int h = (int)map.Shape[2].Fixed;
        int w = (int)map.Shape[3].Fixed;
        if (_preNorm[s] == 1)
        {
            map = Map(LayerNorm(Rows(map, inChannels, h, w), p + "norm"), inChannels, h, w);
        }

        var conv = _b.Emit1("Conv", [_b.Contiguous(map), _b.Weight(p + "conv.weight"), _b.Weight(p + "conv.bias")],
            Attributes.From(("strides", new long[] { _strides[s], _strides[s] }), ("pads", new long[] { _paddings[s], _paddings[s], _paddings[s], _paddings[s] })));
        if (_preNorm[s] == 1)
        {
            return conv;
        }

        int ho = (int)conv.Shape[2].Fixed;
        int wo = (int)conv.Shape[3].Fixed;
        return Map(LayerNorm(Rows(conv, _dims[s], ho, wo), p + "norm"), _dims[s], ho, wo);
    }

    // Depthwise 3x3 convolution with a residual connection over a map.
    private Value DepthwiseResidual(Value map, string p, int c)
    {
        var conv = _b.Emit1("Conv", [_b.Contiguous(map), _b.Weight(p + ".weight"), _b.Weight(p + ".bias")],
            Attributes.From(("strides", new long[] { 1, 1 }), ("pads", new long[] { 1, 1, 1, 1 }), ("groups", (long)c)));
        return _b.Add(conv, map);
    }

    private Value Spatial(Value map, string p, int s, int c, int h, int w)
    {
        map = DepthwiseResidual(map, p + "conv1", c);
        var x = Rows(map, c, h, w);
        x = _b.Add(x, WindowAttention(LayerNorm(x, p + "norm1"), p + "window_attn.", c, _heads[s], h, w));
        map = DepthwiseResidual(Map(x, c, h, w), p + "conv2", c);
        return Map(FeedForward(Rows(map, c, h, w), p), c, h, w);
    }

    private Value Channel(Value map, string p, int s, int c, int h, int w)
    {
        map = DepthwiseResidual(map, p + "conv1", c);
        var x = Rows(map, c, h, w);
        x = _b.Add(x, ChannelAttention(LayerNorm(x, p + "norm1"), p + "channel_attn.", c, _groups[s], h * w));
        map = DepthwiseResidual(Map(x, c, h, w), p + "conv2", c);
        return Map(FeedForward(Rows(map, c, h, w), p), c, h, w);
    }

    // x + MLP(norm2(x)).
    private Value FeedForward(Value x, string p)
    {
        var h = _b.Linear(LayerNorm(x, p + "norm2"), p + "ffn.fc1.weight", p + "ffn.fc1.bias");
        h = _b.Unary(_activation == "gelu_tanh" ? "GeluTanh" : "Gelu", h);
        return _b.Add(x, _b.Linear(h, p + "ffn.fc2.weight", p + "ffn.fc2.bias"));
    }

    // Attention inside non-overlapping windows of zero-padded rows: no position bias, no shift.
    private Value WindowAttention(Value x, string p, int c, int heads, int height, int width)
    {
        int d = c / heads;
        int window = _window;
        int n = window * window;
        int hp = SwinWindows.Padded(height, window);
        int wp = SwinWindows.Padded(width, window);
        int windows = hp / window * (wp / window);
        long batch = (long)windows * heads;
        var (into, back) = SwinWindows.Indices(height, width, window, 0);
        if (hp != height || wp != width)
        {
            x = _b.Concat(0, x, _k.F32($"zero_row_{c}", new Shape(1, c), () => new float[c]));
        }

        var h = _b.Gather(x, _k.I32($"win_{height}_{width}", () => into));
        var qkv = _b.Linear(h, p + "qkv.weight", p + "qkv.bias");
        Value Heads(int part) => _b.Reshape(_b.Permute(_b.Reshape(_b.Slice(qkv, 1, (long)part * c, c), windows, n, heads, d), 0, 2, 1, 3), batch, n, d);
        var scores = _b.Emit1("BatchMatMul", [Heads(0), Heads(1)], Attributes.From(("trans_b", true), ("alpha", 1f / MathF.Sqrt(d))));
        var probs = _b.Emit1("Softmax", [scores]);
        var o = _b.Emit1("BatchMatMul", [probs, Heads(2)]);
        o = _b.Reshape(_b.Contiguous(_b.Permute(_b.Reshape(o, windows, heads, n, d), 0, 2, 1, 3)), (long)windows * n, c);
        o = _b.Linear(o, p + "proj.weight", p + "proj.bias");
        return _b.Gather(o, _k.I32($"unwin_{height}_{width}", () => back));
    }

    // Attention between the channels of each group, over the token axis: softmax(q kᵀ / √N) v with q, k, v [groups, channels per group, tokens].
    private Value ChannelAttention(Value x, string p, int c, int groups, int tokens)
    {
        int d = c / groups;
        var qkv = _b.Linear(x, p + "qkv.weight", p + "qkv.bias");
        Value Part(int part) => _b.Contiguous(_b.Permute(_b.Reshape(_b.Slice(qkv, 1, (long)part * c, c), tokens, groups, d), 1, 2, 0));
        var scores = _b.Emit1("BatchMatMul", [Part(0), Part(1)], Attributes.From(("trans_b", true), ("alpha", 1f / MathF.Sqrt(tokens))));
        var o = _b.Emit1("BatchMatMul", [_b.Emit1("Softmax", [scores]), Part(2)]);
        o = _b.Reshape(_b.Contiguous(_b.Permute(o, 2, 0, 1)), tokens, c);
        return _b.Linear(o, p + "proj.weight", p + "proj.bias");
    }

    // The projector: position and temporal embeddings, the spatial average as an extra first row, the projection and its norm.
    private Value Project(Value map, int c, int h, int w)
    {
        var rows = Rows(map, c, h, w);
        var columns = _b.Gather(_b.Weight("vision.projector.image_position_embed.column_embeddings.weight"), _k.I32($"pos_x_{h}_{w}", () => [.. Enumerable.Range(0, h * w).Select(i => i % w)]));
        var rowEmbeddings = _b.Gather(_b.Weight("vision.projector.image_position_embed.row_embeddings.weight"), _k.I32($"pos_y_{h}_{w}", () => [.. Enumerable.Range(0, h * w).Select(i => i / w)]));
        rows = _b.Add(rows, _b.Concat(1, columns, rowEmbeddings));
        // The temporal embedding of position 0: sin(0) = 0 on the even and cos(0) = 1 on the odd channels.
        rows = _b.Add(rows, _k.F32($"temporal_{c}", new Shape(1, c), () => [.. Enumerable.Range(0, c).Select(i => i % 2 == 0 ? 0f : 1f)]));
        var average = _b.Emit1("Reduce", [rows], Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", true)));
        var tokens = _b.Concat(0, average, rows);
        return LayerNorm(_b.Linear(tokens, "vision.projector.image_projection.weight"), "vision.projector.image_proj_norm");
    }

    private Value LayerNorm(Value x, string name)
        => _b.Emit1("LayerNorm", [x, _b.Weight(name + ".weight"), _b.Weight(name + ".bias")], Attributes.From(("eps", _eps)));

    private Value Rows(Value map, int channels, int height, int width)
        => _b.Reshape(_b.Contiguous(_b.Permute(map, 0, 2, 3, 1)), (long)height * width, channels);

    private Value Map(Value rows, int channels, int height, int width)
        => _b.Contiguous(_b.Permute(_b.Reshape(rows, 1, height, width, channels), 0, 3, 1, 2));
}
