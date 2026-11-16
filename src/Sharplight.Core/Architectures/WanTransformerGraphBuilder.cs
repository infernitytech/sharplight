using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="VideoDiffusionGraphs.Denoise"/>) of a Wan-style video diffusion transformer for one latent
/// size and one prompt length: the latent video is cut into 1×2×2 patches (tokens in (frame, row, column) order), the timestep's
/// sinusoidal embedding gives a time embedding and its six-fold modulation projection, the umT5 hidden states are projected to the
/// model width, and each block runs modulated self-attention (RMS-normed queries and keys over the whole width, RoPE over
/// (frame, row, column)), layer-normed cross-attention to the text and a modulated, gated feed-forward; the output is modulated by the
/// time embedding and projected back to patches. With per-token timesteps (Wan 2.2 TI2V) every token has its own modulation. With an
/// image conditioning (Wan 2.1 I2V / FLF2V) the CLIP image states are embedded (optional position table, LayerNorm, GELU MLP,
/// LayerNorm) and every cross-attention adds an attention of the same queries to the image rows through its own key and value
/// projections.
/// </summary>
/// <remarks>
/// Canonical weights (the original Wan names under <c>transformer.</c>; each <c>.weight</c> with its <c>.bias</c> unless noted):
/// <c>patch_embedding</c> [D, C·pt·ph·pw] (the 3-D convolution kernel flattened in (channel, frame, row, column) order),
/// <c>text_embedding.0|2</c>, <c>time_embedding.0|2</c>, <c>time_projection.1</c> [6D, D]; per block <c>blocks.{i}.</c>
/// <c>self_attn.q|k|v|o</c>, <c>self_attn.norm_q|norm_k.weight</c> [D], <c>norm3</c> (the cross-attention input norm, when present),
/// <c>cross_attn.q|k|v|o</c>, <c>cross_attn.norm_q|norm_k.weight</c>, <c>ffn.0|2</c>, <c>modulation</c> f32 [6D] (shift, scale and
/// gate of the self-attention, then of the feed-forward); <c>head.head</c> [pt·ph·pw·C, D] (output features in (frame, row,
/// column, channel) order) and <c>head.modulation</c> f32 [2D] (shift, scale). Image conditioning: <c>img_emb.proj.0|4</c> (LayerNorms),
/// <c>img_emb.proj.1|3</c> (the MLP), optional <c>img_emb.emb_pos</c> f32 [image rows, image width] (FLF2V), per block
/// <c>cross_attn.k_img|v_img</c> and <c>cross_attn.norm_k_img.weight</c>.
/// </remarks>
internal sealed class WanTransformerGraphBuilder
{
    private readonly GraphBuilder _b;
    private readonly string _prefix;
    private readonly WanHyperParameters _hp;
    private readonly long _frames;
    private readonly long _rows;
    private readonly long _columns;
    private readonly int _textRows;
    private readonly int _tokens;
    private Value _positions = null!;
    private Value _frequencyFactors = null!;

    private WanTransformerGraphBuilder(Func<string, WeightInfo?> weights, WanHyperParameters hp, long latentFrames, long latentHeight, long latentWidth,
        int textRows, string prefix, string name)
    {
        if (hp.PatchSize[0] != 1)
        {
            throw new NotSupportedException($"Wan transformer: temporal patch size {hp.PatchSize[0]} is not supported (only 1).");
        }

        _b = new GraphBuilder(name, weights);
        _prefix = prefix;
        _hp = hp;
        _frames = latentFrames;
        _rows = latentHeight / hp.PatchSize[1];
        _columns = latentWidth / hp.PatchSize[2];
        _textRows = textRows;
        _tokens = (int)(_frames * _rows * _columns);
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="latentFrames">Latent frames.</param>
    /// <param name="latentHeight">Latent height (a multiple of the patch height).</param>
    /// <param name="latentWidth">Latent width (a multiple of the patch width).</param>
    /// <param name="textRows">Tokens of the text conditioning.</param>
    /// <param name="prefix">Canonical weight prefix (<c>transformer.</c>, or <c>transformer_2.</c> for a second expert).</param>
    /// <param name="name">Graph name.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, WanHyperParameters hp, long latentFrames, long latentHeight, long latentWidth, int textRows,
        string prefix = "transformer.", string name = VideoDiffusionGraphs.Denoise)
        => new WanTransformerGraphBuilder(weights, hp, latentFrames, latentHeight, latentWidth, textRows, prefix, name).Build();

    /// <summary>Tokens of a latent video of that size.</summary>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="latentFrames">Latent frames.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    public static int TokenCount(WanHyperParameters hp, long latentFrames, long latentHeight, long latentWidth)
        => (int)(latentFrames / hp.PatchSize[0] * (latentHeight / hp.PatchSize[1]) * (latentWidth / hp.PatchSize[2]));

    private ComputeGraph Build()
    {
        int d = _hp.HiddenSize;
        long latentHeight = _rows * _hp.PatchSize[1];
        long latentWidth = _columns * _hp.PatchSize[2];
        var latent = _b.Input(VideoDiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(_hp.InChannels, _frames, latentHeight, latentWidth));
        var timestep = _b.Input(VideoDiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(_hp.TokenTimesteps ? _tokens : 1));
        var context = _b.Input(VideoDiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(_textRows, _hp.TextDim));
        var imageContext = _hp.ImageDim > 0
            ? _b.Input(VideoDiffusionGraphs.ImageContext, InputRole.Other, DataType.F32, new Shape(_hp.ImageTokens, _hp.ImageDim))
            : null;
        DeclarePositions();

        Value timeEmbedding;
        Value modulation;
        Value text;
        Value? image = null;
        using (_b.Group("conditioning"))
        {
            timeEmbedding = Linear(_b.Silu(Linear(Sinusoids(timestep), _prefix + "time_embedding.0")), _prefix + "time_embedding.2");
            modulation = Linear(_b.Silu(timeEmbedding), _prefix + "time_projection.1");
            text = Linear(_b.Unary("GeluTanh", Linear(context, _prefix + "text_embedding.0")), _prefix + "text_embedding.2");
            if (imageContext is not null)
            {
                image = ImageEmbedding(imageContext);
            }
        }

        Value x;
        using (_b.Group("input"))
        {
            x = Linear(Patchify(latent), _prefix + "patch_embedding");
        }

        for (int i = 0; i < _hp.LayerCount; i++)
        {
            using var _ = _b.Group($"blocks.{i}");
            x = Block(x, text, image, modulation, $"{_prefix}blocks.{i}.");
        }

        using (_b.Group("output"))
        {
            var table = _b.Weight(_prefix + "head.modulation");
            var shift = _b.Add(_b.Reshape(_b.Slice(table, 0, 0, d), 1, d), timeEmbedding);
            var scale = _b.Add(_b.Reshape(_b.Slice(table, 0, d, d), 1, d), timeEmbedding);
            var projected = Linear(Modulate(x, shift, scale), _prefix + "head.head");
            _b.Output(VideoDiffusionGraphs.Prediction, Unpatchify(projected));
        }

        return _b.Build();
    }

    /// <summary>The constant RoPE positions [tokens, 3] = (frame, row, column) and the per-axis frequency divisors.</summary>
    private void DeclarePositions()
    {
        var positions = new int[_tokens * 3];
        long perFrame = _rows * _columns;
        for (int t = 0; t < _tokens; t++)
        {
            positions[t * 3] = (int)(t / perFrame);
            positions[t * 3 + 1] = (int)(t % perFrame / _columns);
            positions[t * 3 + 2] = (int)(t % _columns);
        }

        _positions = _b.Constant("rope_positions", DataType.I32, new Shape(_tokens, 3), MemoryMarshal.AsBytes(positions.AsSpan()).ToArray());
        _frequencyFactors = AxialRope.FrequencyFactors(_b, "rope_frequency_factors", _hp.HeadDim, _hp.AxesDim, _hp.RopeTheta);
    }

    /// <summary>One transformer block over the token rows [tokens, D].</summary>
    private Value Block(Value x, Value text, Value? image, Value timeModulation, string p)
    {
        int d = _hp.HiddenSize;
        var mod = _b.Add(timeModulation, _b.Reshape(_b.Weight(p + "modulation"), 1, 6L * d));
        var attended = SelfAttention(Modulate(x, Chunk(mod, 0), Chunk(mod, 1)), p + "self_attn");
        x = _b.Add(x, _b.Mul(attended, Chunk(mod, 2)));
        var crossInput = _hp.CrossAttentionNorm ? DiffusionBlocks.LayerNorm(_b, x, p + "norm3", _hp.NormEps) : x;
        x = _b.Add(x, CrossAttention(crossInput, text, image, p + "cross_attn"));
        var hidden = Modulate(x, Chunk(mod, 3), Chunk(mod, 4));
        var feedForward = Linear(_b.Unary("GeluTanh", Linear(hidden, p + "ffn.0")), p + "ffn.2");
        return _b.Add(x, _b.Mul(feedForward, Chunk(mod, 5)));
    }

    private Value SelfAttention(Value x, string p)
    {
        var q = Rotate(Heads(_b.RmsNorm(Linear(x, p + ".q"), _b.Weight(p + ".norm_q.weight"), _hp.NormEps)));
        var k = Rotate(Heads(_b.RmsNorm(Linear(x, p + ".k"), _b.Weight(p + ".norm_k.weight"), _hp.NormEps)));
        var v = Heads(Linear(x, p + ".v"));
        return Linear(Attend(q, k, v, _tokens), p + ".o");
    }

    private Value CrossAttention(Value x, Value text, Value? image, string p)
    {
        var q = Heads(_b.RmsNorm(Linear(x, p + ".q"), _b.Weight(p + ".norm_q.weight"), _hp.NormEps));
        var k = Heads(_b.RmsNorm(Linear(text, p + ".k"), _b.Weight(p + ".norm_k.weight"), _hp.NormEps));
        var v = Heads(Linear(text, p + ".v"));
        var attended = Attend(q, k, v, _tokens);
        if (image is not null)
        {
            var imageKeys = Heads(_b.RmsNorm(Linear(image, p + ".k_img"), _b.Weight(p + ".norm_k_img.weight"), _hp.NormEps));
            var imageValues = Heads(Linear(image, p + ".v_img"));
            attended = _b.Add(attended, Attend(q, imageKeys, imageValues, _tokens));
        }

        return Linear(attended, p + ".o");
    }

    /// <summary>The image conditioning [image rows, D]: position table (FLF2V), LayerNorm, Linear, GELU, Linear, LayerNorm.</summary>
    private Value ImageEmbedding(Value states)
    {
        string p = _prefix + "img_emb.";
        if (_b.HasWeight(p + "emb_pos"))
        {
            states = _b.Add(states, _b.Reshape(_b.Weight(p + "emb_pos"), _hp.ImageTokens, _hp.ImageDim));
        }

        var normed = DiffusionBlocks.LayerNorm(_b, states, p + "proj.0", 1e-5f);
        var hidden = Linear(_b.Unary("Gelu", Linear(normed, p + "proj.1")), p + "proj.3");
        return DiffusionBlocks.LayerNorm(_b, hidden, p + "proj.4", 1e-5f);
    }

    private Value Rotate(Value heads) => _b.Emit1("Rope", [heads, _positions, _frequencyFactors], AxialRope.Attributes(_hp.HeadDim, _hp.AxesDim, _hp.RopeTheta));

    /// <summary>Bidirectional attention; returns rows [query rows, D].</summary>
    private Value Attend(Value q, Value k, Value v, int queryRows)
    {
        // f16 products with f32 accumulation (tensor-core flash attention): Wan is trained and run in bf16.
        var attended = _b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(_hp.HeadDim)), ("causal", false),
            (StandardOps.F16ProductsAttribute, true)));
        return _b.Reshape(attended, queryRows, _hp.HiddenSize);
    }

    private Value Heads(Value x) => _b.Reshape(_b.Contiguous(x), x.Shape[0].Fixed, _hp.HeadCount, _hp.HeadDim);

    /// <summary>LayerNorm without affine parameters, then x · (1 + scale) + shift (per-token or broadcast rows [S, D]).</summary>
    private Value Modulate(Value x, Value shift, Value scale)
    {
        var normed = _b.Emit1("LayerNorm", [x], Attributes.From(("eps", _hp.NormEps)));
        return _b.Add(_b.Mul(normed, _b.Scale(scale, 1f, 1f)), shift);
    }

    /// <summary>Chunk <paramref name="index"/> [S, D] of a six-fold modulation [S, 6D].</summary>
    private Value Chunk(Value modulation, int index) => _b.Slice(modulation, 1, (long)index * _hp.HiddenSize, _hp.HiddenSize);

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", _b.HasWeight(module + ".bias") ? module + ".bias" : null);

    /// <summary>
    /// The sinusoidal embedding of the timesteps [S] into [S, F] channels, cosines first, frequencies 10000^(−i / (F/2)) (diffusers'
    /// <c>Timesteps(F, flip_sin_to_cos=True, downscale_freq_shift=0)</c>).
    /// </summary>
    private Value Sinusoids(Value timesteps)
    {
        int half = _hp.FrequencyDim / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)Math.Exp(-Math.Log(10000) * i / half);
        }

        var table = _b.Constant("time_frequencies", DataType.F32, new Shape(1, half), MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        long rows = timesteps.Shape[0].Fixed;
        var arguments = _b.Mul(_b.Reshape(timesteps, rows, 1), table);
        return _b.Concat(1, _b.Unary("Cos", arguments), _b.Unary("Sin", arguments));
    }

    /// <summary>Latents [C, T, h, w] → patch rows [T · h/2 · w/2, C·2·2], features ordered (channel, row in patch, column in patch).</summary>
    private Value Patchify(Value latent)
    {
        int ph = _hp.PatchSize[1];
        int pw = _hp.PatchSize[2];
        int c = _hp.InChannels;
        var split = _b.Reshape(latent, c, _frames, _rows, ph, _columns, pw);
        var ordered = _b.Contiguous(_b.Permute(split, 1, 2, 4, 0, 3, 5));
        return _b.Reshape(ordered, _tokens, (long)c * ph * pw);
    }

    /// <summary>Output rows [tokens, 2·2·C] with features ordered (row in patch, column in patch, channel) → latents [C, T, h, w].</summary>
    private Value Unpatchify(Value patches)
    {
        int ph = _hp.PatchSize[1];
        int pw = _hp.PatchSize[2];
        int c = _hp.OutChannels;
        var split = _b.Reshape(patches, _frames, _rows, _columns, ph, pw, c);
        var ordered = _b.Contiguous(_b.Permute(split, 5, 0, 1, 3, 2, 4));
        return _b.Reshape(ordered, c, _frames, _rows * ph, _columns * pw);
    }
}
