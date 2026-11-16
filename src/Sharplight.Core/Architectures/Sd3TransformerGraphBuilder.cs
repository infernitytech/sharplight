using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="DiffusionGraphs.Denoise"/>) of an SD3-style MMDiT for one latent size and one conditioning
/// length: the latents are cut into p×p patches (tokens) and embedded, plus the 2D sin-cos position embeddings cropped from the centre
/// of a fixed grid; the timestep and the pooled text embedding form the modulation vector; every block attends jointly over the image
/// and text tokens (image first) with separate weights per stream, SD3.5's first blocks add a second, image-only attention, and the last
/// block only updates the image stream; the image tokens are projected back to latents. One image per run (a fixed batch of 1).
/// </summary>
/// <remarks>
/// Canonical weights (each <c>.weight</c> with its <c>.bias</c>, except the norm scales): <c>transformer.patch_embed</c> [D, C·p²]
/// (patch features ordered (channel, row in patch, column in patch)), <c>transformer.pos_embed</c> [G·G, D] (the grid of
/// <see cref="Sd3HyperParameters.PositionGrid"/>), <c>transformer.txt_in</c> [D, context], <c>transformer.time_in.in_layer|out_layer</c>,
/// <c>transformer.vector_in.in_layer|out_layer</c>; per block <c>transformer.joint.{i}.</c> <c>img_mod</c> [6D, D] (shift, scale, gate of
/// the attention, then of the MLP; [9D, D] with the second attention's shift, scale, gate), <c>txt_mod</c> [6D, D] ([2D, D] (shift,
/// scale) in the last block), <c>{img,txt}_attn.qkv</c> [3D, D], optional <c>{img,txt}_attn.norm_q|norm_k.weight</c> [head dim],
/// <c>img_attn.proj</c>, <c>txt_attn.proj</c> (not in the last block), <c>{img,txt}_mlp.up|down</c> (no <c>txt_mlp</c> in the last
/// block), with the second attention <c>img_attn2.qkv</c>, <c>img_attn2.norm_q|norm_k.weight</c>, <c>img_attn2.proj</c>;
/// <c>transformer.final.mod</c> [2D, D] (shift, scale), <c>transformer.final.linear</c> [p²·C, D] (output features ordered (row in
/// patch, column in patch, channel)).
/// </remarks>
internal sealed class Sd3TransformerGraphBuilder
{
    private const string P = "transformer.";
    private const float NormEps = 1e-6f;
    private const int FrequencyChannels = 256;

    private readonly GraphBuilder _b;
    private readonly Sd3HyperParameters _hp;
    private readonly int _textRows;
    private readonly int _imageRows;
    private readonly long _latentHeight;
    private readonly long _latentWidth;

    private Sd3TransformerGraphBuilder(Func<string, WeightInfo?> weights, Sd3HyperParameters hp, long latentHeight, long latentWidth, int textRows)
    {
        _b = new GraphBuilder(DiffusionGraphs.Denoise, weights);
        _hp = hp;
        _textRows = textRows;
        _latentHeight = latentHeight;
        _latentWidth = latentWidth;
        _imageRows = (int)(latentHeight / hp.PatchSize * (latentWidth / hp.PatchSize));
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="latentHeight">Latent height (a multiple of the patch size).</param>
    /// <param name="latentWidth">Latent width (a multiple of the patch size).</param>
    /// <param name="textRows">Tokens of the text conditioning (CLIP and T5 rows together).</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, Sd3HyperParameters hp, long latentHeight, long latentWidth, int textRows)
        => new Sd3TransformerGraphBuilder(weights, hp, latentHeight, latentWidth, textRows).Build();

    private ComputeGraph Build()
    {
        int d = _hp.HiddenSize;
        long rows = _latentHeight / _hp.PatchSize;
        long columns = _latentWidth / _hp.PatchSize;
        if (rows > _hp.PositionGrid || columns > _hp.PositionGrid)
        {
            throw new ArgumentException($"The image is {columns}×{rows} tokens; this SD3 transformer's position embeddings cover at most {_hp.PositionGrid}×{_hp.PositionGrid} "
                + $"({_hp.PositionGrid * _hp.PatchSize * 8} pixels).");
        }

        var latent = _b.Input(DiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(1, _hp.LatentChannels, _latentHeight, _latentWidth));
        var timestep = _b.Input(DiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(1));
        var context = _b.Input(DiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(1, _textRows, _hp.ContextDim));
        var pooled = _b.Input(DiffusionGraphs.Pooled, InputRole.Other, DataType.F32, new Shape(1, _hp.PooledDim));

        Value vector;
        using (_b.Group("conditioning"))
        {
            vector = _b.Add(Mlp(Sinusoids(timestep), P + "time_in"), Mlp(pooled, P + "vector_in"));
        }

        var modulationInput = _b.Silu(vector);
        Value image;
        Value text;
        using (_b.Group("input"))
        {
            image = _b.Add(Linear(Patchify(latent), P + "patch_embed"), PositionEmbeddings(rows, columns));
            text = Linear(_b.Reshape(context, _textRows, _hp.ContextDim), P + "txt_in");
        }

        for (int i = 0; i < _hp.Layers; i++)
        {
            using var _ = _b.Group($"joint.{i}");
            (image, text) = JointBlock(image, text, modulationInput, $"{P}joint.{i}.", last: i == _hp.Layers - 1, dual: i < _hp.DualAttentionLayers);
        }

        using (_b.Group("output"))
        {
            var final = Linear(modulationInput, P + "final.mod");
            var normed = Modulate(image, Chunk(final, 0, d), Chunk(final, 1, d));
            _b.Output(DiffusionGraphs.Prediction, Unpatchify(Linear(normed, P + "final.linear")));
        }

        return _b.Build();
    }

    /// <summary>The position embeddings of the image tokens [rows · columns, D]: the centre crop of the stored grid.</summary>
    private Value PositionEmbeddings(long rows, long columns)
    {
        int grid = _hp.PositionGrid;
        long top = (grid - rows) / 2;
        long left = (grid - columns) / 2;
        var table = _b.Reshape(_b.Weight(P + "pos_embed"), grid, grid, _hp.HiddenSize);
        var crop = _b.Slice(_b.Slice(table, 0, top, rows), 1, left, columns);
        var embeddings = _b.Reshape(_b.Contiguous(crop), rows * columns, _hp.HiddenSize);
        return _b.Emit1("Cast", [embeddings], Attributes.From(("type", "f32")));
    }

    /// <summary>
    /// A joint-attention block: modulated pre-norm attention over the image and text tokens together (each stream with its own
    /// projections), the optional second image-only attention from the same normed input, and modulated MLPs. The last block only
    /// projects the text stream to keys and values (its text output is unused).
    /// </summary>
    private (Value Image, Value Text) JointBlock(Value image, Value text, Value modulationInput, string p, bool last, bool dual)
    {
        int d = _hp.HiddenSize;
        var imageMod = Linear(modulationInput, p + "img_mod");
        var textMod = Linear(modulationInput, p + "txt_mod");
        var normedImage = _b.Emit1("LayerNorm", [image], Attributes.From(("eps", NormEps)));
        var normedText = _b.Emit1("LayerNorm", [text], Attributes.From(("eps", NormEps)));
        var (iq, ik, iv) = Projections(Affine(normedImage, Chunk(imageMod, 0, d), Chunk(imageMod, 1, d)), p + "img_attn", _imageRows);
        var (tq, tk, tv) = Projections(Affine(normedText, Chunk(textMod, 0, d), Chunk(textMod, 1, d)), p + "txt_attn", _textRows);
        var attended = Attend(_b.Concat(0, iq, tq), _b.Concat(0, ik, tk), _b.Concat(0, iv, tv), _imageRows + _textRows);
        var imageAttention = _b.Slice(attended, 0, 0, _imageRows);
        image = _b.Add(image, _b.Mul(Chunk(imageMod, 2, d), Linear(imageAttention, p + "img_attn.proj")));
        if (dual)
        {
            var (q2, k2, v2) = Projections(Affine(normedImage, Chunk(imageMod, 6, d), Chunk(imageMod, 7, d)), p + "img_attn2", _imageRows);
            image = _b.Add(image, _b.Mul(Chunk(imageMod, 8, d), Linear(Attend(q2, k2, v2, _imageRows), p + "img_attn2.proj")));
        }

        image = _b.Add(image, _b.Mul(Chunk(imageMod, 5, d), FeedForward(Modulate(image, Chunk(imageMod, 3, d), Chunk(imageMod, 4, d)), p + "img_mlp")));
        if (last)
        {
            return (image, text);
        }

        var textAttention = _b.Slice(attended, 0, _imageRows, _textRows);
        text = _b.Add(text, _b.Mul(Chunk(textMod, 2, d), Linear(textAttention, p + "txt_attn.proj")));
        text = _b.Add(text, _b.Mul(Chunk(textMod, 5, d), FeedForward(Modulate(text, Chunk(textMod, 3, d), Chunk(textMod, 4, d)), p + "txt_mlp")));
        return (image, text);
    }

    /// <summary>The query, key and value heads of one stream, queries and keys RMS-normed when the model has the norms.</summary>
    private (Value Q, Value K, Value V) Projections(Value x, string p, int rows)
    {
        int d = _hp.HiddenSize;
        var qkv = Linear(x, p + ".qkv");
        var q = Heads(_b.Contiguous(_b.Slice(qkv, 1, 0, d)), rows);
        var k = Heads(_b.Contiguous(_b.Slice(qkv, 1, d, d)), rows);
        var v = Heads(_b.Contiguous(_b.Slice(qkv, 1, 2 * d, d)), rows);
        if (_b.HasWeight(p + ".norm_q.weight"))
        {
            q = _b.RmsNorm(q, _b.Weight(p + ".norm_q.weight"), NormEps);
            k = _b.RmsNorm(k, _b.Weight(p + ".norm_k.weight"), NormEps);
        }

        return (q, k, v);
    }

    /// <summary>Bidirectional attention; returns rows [T, D].</summary>
    private Value Attend(Value q, Value k, Value v, int rows)
    {
        // f16 products with f32 accumulation (tensor-core flash attention) when queries and keys are RMS-normed (SD3.5, as FLUX): their
        // scores stay small. SD3's unnormed queries and keys give large scores whose f16 rounding costs ~1 % of the output.
        var attended = _b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(_hp.HeadDim)), ("causal", false),
            (StandardOps.F16ProductsAttribute, _hp.QkNorm)));
        return _b.Reshape(attended, rows, _hp.HiddenSize);
    }

    private Value Heads(Value x, int rows) => _b.Reshape(x, rows, _hp.HeadCount, _hp.HeadDim);

    /// <summary>LayerNorm without affine parameters, then x · (1 + scale) + shift (per-feature vectors [1, D]).</summary>
    private Value Modulate(Value x, Value shift, Value scale) => Affine(_b.Emit1("LayerNorm", [x], Attributes.From(("eps", NormEps))), shift, scale);

    private Value Affine(Value normed, Value shift, Value scale) => _b.Add(_b.Mul(normed, _b.Scale(scale, 1f, 1f)), shift);

    /// <summary>Chunk <paramref name="index"/> of width <paramref name="width"/> of a modulation vector [1, k·D].</summary>
    private Value Chunk(Value modulation, int index, int width) => _b.Slice(modulation, 1, (long)index * width, width);

    private Value FeedForward(Value x, string p) => Linear(_b.Unary("GeluTanh", Linear(x, p + ".up")), p + ".down");

    /// <summary>Two linear layers with a SiLU between (the timestep and pooled-text embedders).</summary>
    private Value Mlp(Value x, string p) => Linear(_b.Silu(Linear(x, p + ".in_layer")), p + ".out_layer");

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", _b.HasWeight(module + ".bias") ? module + ".bias" : null);

    /// <summary>
    /// The sinusoidal embedding of the timestep [1] into 256 channels, cosines first, frequencies 10000^(−i / 128) (diffusers'
    /// <c>Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0)</c>).
    /// </summary>
    private Value Sinusoids(Value values)
    {
        int half = FrequencyChannels / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)Math.Exp(-Math.Log(10000) * i / half);
        }

        var table = _b.Constant("time_frequencies", DataType.F32, new Shape(1, half), MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var arguments = _b.Mul(_b.Reshape(values, 1, 1), table);
        return _b.Concat(1, _b.Unary("Cos", arguments), _b.Unary("Sin", arguments));
    }

    /// <summary>Latents [1, C, h, w] → patch rows [h/p · w/p, C·p·p], features ordered (channel, row in patch, column in patch).</summary>
    private Value Patchify(Value latent)
    {
        int p = _hp.PatchSize;
        int c = _hp.LatentChannels;
        long rows = _latentHeight / p;
        long columns = _latentWidth / p;
        var split = _b.Reshape(latent, c, rows, p, columns, p);
        var ordered = _b.Contiguous(_b.Permute(split, 1, 3, 0, 2, 4));
        return _b.Reshape(ordered, rows * columns, (long)c * p * p);
    }

    /// <summary>Patch rows [h/p · w/p, p·p·C] (features ordered (row in patch, column in patch, channel)) → latents [1, C, h, w].</summary>
    private Value Unpatchify(Value patches)
    {
        int p = _hp.PatchSize;
        int c = _hp.LatentChannels;
        long rows = _latentHeight / p;
        long columns = _latentWidth / p;
        var split = _b.Reshape(patches, rows, columns, p, p, c);
        var ordered = _b.Contiguous(_b.Permute(split, 4, 0, 2, 1, 3));
        return _b.Reshape(ordered, 1, c, _latentHeight, _latentWidth);
    }
}
