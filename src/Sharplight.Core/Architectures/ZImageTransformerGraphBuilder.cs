using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="DiffusionGraphs.Denoise"/>) of a Z-Image single-stream DiT for one latent size: the latents are
/// cut into p×p patches (tokens, padded to a multiple of 32 with a learned pad token) and refined by modulated blocks; the caption
/// features (a language model's hidden states, the prompt's tokens visible through <see cref="DiffusionGraphs.ContextMask"/>, padded to
/// a multiple of 32 with a learned pad token, the rest of the fixed caption rows masked out) are refined by unmodulated blocks; joint
/// blocks run over [image, caption]; the image tokens are projected back to latents. RoPE positions: caption token i at (1 + i, 0, 0),
/// image token at (padded caption length + 1, row, column), image padding at the origin; the caption length is computed in the graph
/// from the mask, so one graph serves every prompt. The denoiser takes the model timestep σ · 1000 and returns the flow velocity
/// (Z-Image's own convention, timestep 1 − σ and the negated velocity, stays inside). One image per run.
/// </summary>
/// <remarks>
/// Canonical weights (bias only where named): <c>transformer.x_embedder.weight|bias</c> [D, p²·C] (patch features ordered (row in
/// patch, column in patch, channel)), <c>transformer.x_pad_token</c> / <c>transformer.cap_pad_token</c> [1, D],
/// <c>transformer.t_embedder.in_layer|out_layer.weight|bias</c>, <c>transformer.cap_embedder.norm.weight</c>,
/// <c>transformer.cap_embedder.linear.weight|bias</c>; per block <c>transformer.{noise_refiner|context_refiner|layers}.{i}.</c>
/// <c>attention.qkv.weight</c> [3D, D], <c>attention.out.weight</c>, <c>attention.norm_q|norm_k.weight</c> [head dim],
/// <c>attention_norm1|attention_norm2|ffn_norm1|ffn_norm2.weight</c>, <c>ffn.w1|w2|w3.weight</c> (SwiGLU: w2(silu(w1·x) ⊙ w3·x)) and,
/// modulated, <c>mod.weight|bias</c> [4D, M] (scale and tanh gate of the attention, then of the feed-forward);
/// <c>transformer.final.mod.weight|bias</c> [D, M], <c>transformer.final.linear.weight|bias</c> [p²·C, D].
/// </remarks>
internal sealed class ZImageTransformerGraphBuilder
{
    private const string P = "transformer.";
    private const int FrequencyChannels = 256;
    private const int SequenceMultiple = 32;

    private readonly GraphBuilder _b;
    private readonly ZImageHyperParameters _hp;
    private readonly int _captionRows;
    private readonly int _imageTokens;
    private readonly int _imageRows;
    private readonly long _latentHeight;
    private readonly long _latentWidth;
    private Value _frequencyFactors = null!;

    private ZImageTransformerGraphBuilder(Func<string, WeightInfo?> weights, ZImageHyperParameters hp, long latentHeight, long latentWidth, int captionRows)
    {
        _b = new GraphBuilder(DiffusionGraphs.Denoise, weights);
        _hp = hp;
        _captionRows = captionRows;
        _latentHeight = latentHeight;
        _latentWidth = latentWidth;
        _imageTokens = (int)(latentHeight / hp.PatchSize * (latentWidth / hp.PatchSize));
        _imageRows = RoundUp(_imageTokens);
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="latentHeight">Latent height (a multiple of the patch size).</param>
    /// <param name="latentWidth">Latent width (a multiple of the patch size).</param>
    /// <param name="captionRows">Caption rows (the text encoder's prompt length, a multiple of 32).</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, ZImageHyperParameters hp, long latentHeight, long latentWidth, int captionRows)
        => new ZImageTransformerGraphBuilder(weights, hp, latentHeight, latentWidth, captionRows).Build();

    private static int RoundUp(int rows) => (rows + SequenceMultiple - 1) / SequenceMultiple * SequenceMultiple;

    private ComputeGraph Build()
    {
        if (_captionRows % SequenceMultiple != 0)
        {
            throw new ArgumentException($"Z-Image: the caption rows ({_captionRows}) must be a multiple of {SequenceMultiple}.");
        }

        var latent = _b.Input(DiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(1, _hp.LatentChannels, _latentHeight, _latentWidth));
        var timestep = _b.Input(DiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(1));
        var context = _b.Input(DiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(1, _captionRows, _hp.CaptionDim));
        var mask = _b.Input(DiffusionGraphs.ContextMask, InputRole.Other, DataType.F32, new Shape(1, _captionRows));
        DeclareFrequencyFactors();

        Value adaln;
        Value captionBias;
        Value imagePositions;
        Value captionPositions;
        using (_b.Group("conditioning"))
        {
            // Z-Image's timestep is 1 − σ = (1000 − t) / 1000, scaled by the model's time scale.
            var time = _b.Scale(timestep, -_hp.TimeScale / 1000f, _hp.TimeScale);
            adaln = Mlp(Sinusoids(time), P + "t_embedder");
            // The padded caption length c = 32 · ⌈n / 32⌉ of the n visible tokens; caption keys at or after c are masked.
            var visible = _b.Emit1("ReduceSum", [mask]);
            var padded = _b.Scale(_b.Unary("Ceil", _b.Scale(visible, 1f / SequenceMultiple)), SequenceMultiple);
            var iota = new float[_captionRows];
            for (int i = 0; i < _captionRows; i++)
            {
                iota[i] = i;
            }

            var indices = _b.Constant("caption_indices", DataType.F32, new Shape(1, _captionRows), MemoryMarshal.AsBytes(iota.AsSpan()).ToArray());
            // 1 for i < c, else 0 (integral values): min(max(c − i, 0), 1).
            var inside = _b.Scale(_b.Unary("Relu", _b.Scale(_b.Unary("Relu", _b.Sub(padded, indices)), -1f, 1f)), -1f, 1f);
            captionBias = _b.Reshape(_b.Scale(inside, 1e9f, -1e9f), _captionRows);
            (imagePositions, captionPositions) = Positions(padded);
        }

        Value image;
        Value caption;
        var zeros = _b.Constant("image_key_bias", DataType.F32, new Shape(_imageRows), new byte[_imageRows * sizeof(float)]);
        using (_b.Group("input"))
        {
            image = Linear(Patchify(latent), P + "x_embedder");
            if (_imageRows > _imageTokens)
            {
                image = _b.Concat(0, image, Repeat(_b.Weight(P + "x_pad_token"), _imageRows - _imageTokens, "image_padding"));
            }

            var captionFeatures = Linear(_b.RmsNorm(_b.Reshape(context, _captionRows, _hp.CaptionDim), _b.Weight(P + "cap_embedder.norm.weight"), _hp.NormEps),
                P + "cap_embedder.linear");
            // Rows after the prompt become the caption pad token: m · x + (1 − m) · pad.
            var rowMask = _b.Reshape(mask, _captionRows, 1);
            var padToken = Repeat(_b.Weight(P + "cap_pad_token"), _captionRows, "caption_padding");
            caption = _b.Add(_b.Mul(captionFeatures, rowMask), _b.Mul(padToken, _b.Scale(rowMask, -1f, 1f)));
        }

        for (int i = 0; i < _hp.RefinerLayers; i++)
        {
            using var _ = _b.Group($"noise_refiner.{i}");
            image = Block(image, adaln, imagePositions, null, _imageRows, $"{P}noise_refiner.{i}.");
        }

        for (int i = 0; i < _hp.RefinerLayers; i++)
        {
            using var _ = _b.Group($"context_refiner.{i}");
            caption = Block(caption, null, captionPositions, captionBias, _captionRows, $"{P}context_refiner.{i}.");
        }

        var x = _b.Concat(0, image, caption);
        var positions = _b.Concat(0, imagePositions, captionPositions);
        var bias = _b.Concat(0, zeros, captionBias);
        for (int i = 0; i < _hp.Layers; i++)
        {
            using var _ = _b.Group($"layers.{i}");
            x = Block(x, adaln, positions, bias, _imageRows + _captionRows, $"{P}layers.{i}.");
        }

        using (_b.Group("output"))
        {
            var scale = _b.Scale(Linear(_b.Silu(adaln), P + "final.mod"), 1f, 1f);
            var normed = _b.Mul(_b.Emit1("LayerNorm", [_b.Slice(x, 0, 0, _imageTokens)], Attributes.From(("eps", 1e-6f))), scale);
            // Z-Image predicts the negated velocity.
            _b.Output(DiffusionGraphs.Prediction, _b.Scale(Unpatchify(Linear(normed, P + "final.linear")), -1f));
        }

        return _b.Build();
    }

    /// <summary>
    /// The RoPE positions i32: image rows [image rows, 3] at (c + 1, row, column) for the tokens (c the padded caption length, [1, 1]) and
    /// the origin for the padding; caption rows [caption rows, 3] at (1 + i, 0, 0).
    /// </summary>
    private (Value Image, Value Caption) Positions(Value padded)
    {
        long columns = _latentWidth / _hp.PatchSize;
        var fixedPart = new float[_imageRows * 3];
        var indicator = new float[_imageRows * 3];
        for (int t = 0; t < _imageTokens; t++)
        {
            fixedPart[t * 3 + 1] = t / columns;
            fixedPart[t * 3 + 2] = t % columns;
            indicator[t * 3] = 1f;
        }

        var baseValues = _b.Constant("image_positions", DataType.F32, new Shape(_imageRows, 3), MemoryMarshal.AsBytes(fixedPart.AsSpan()).ToArray());
        var axis0 = _b.Constant("image_position_axis0", DataType.F32, new Shape(_imageRows, 3), MemoryMarshal.AsBytes(indicator.AsSpan()).ToArray());
        var image = _b.Emit1("Cast", [_b.Add(baseValues, _b.Mul(axis0, _b.Scale(padded, 1f, 1f)))], Attributes.From(("type", "i32")));
        var captionValues = new int[_captionRows * 3];
        for (int i = 0; i < _captionRows; i++)
        {
            captionValues[i * 3] = 1 + i;
        }

        var caption = _b.Constant("caption_positions", DataType.I32, new Shape(_captionRows, 3), MemoryMarshal.AsBytes(captionValues.AsSpan()).ToArray());
        return (image, caption);
    }

    /// <summary>The per-frequency divisors that turn the Rope op's single series into Z-Image's per-axis series θ^(−2j / axis dims).</summary>
    private void DeclareFrequencyFactors()
    {
        int headDim = _hp.HeadDim;
        var factors = new float[headDim / 2];
        int pair = 0;
        foreach (int axisDims in _hp.AxesDim)
        {
            for (int j = 0; j < axisDims / 2; j++, pair++)
            {
                double op = Math.Pow(_hp.RopeTheta, -2.0 * pair / headDim);
                double own = Math.Pow(_hp.RopeTheta, -2.0 * j / axisDims);
                factors[pair] = (float)(op / own);
            }
        }

        if (pair != headDim / 2)
        {
            throw new InvalidDataException($"Z-Image RoPE axes {string.Join("+", _hp.AxesDim)} do not cover the head width {headDim}.");
        }

        _frequencyFactors = _b.Constant("rope_frequency_factors", DataType.F32, new Shape(headDim / 2), MemoryMarshal.AsBytes(factors.AsSpan()).ToArray());
    }

    /// <summary>
    /// A block: RMS-normed (sandwich) attention and SwiGLU feed-forward, with the modulation's scales and tanh gates when
    /// <paramref name="adaln"/> is given.
    /// </summary>
    private Value Block(Value x, Value? adaln, Value positions, Value? keyBias, int rows, string p)
    {
        int d = _hp.HiddenSize;
        Value? mod = adaln is null ? null : Linear(adaln, p + "mod");
        Value Chunk(int index) => _b.Slice(mod!, 1, (long)index * d, d);
        var attentionInput = Norm(x, p + "attention_norm1");
        if (mod is not null)
        {
            attentionInput = _b.Mul(attentionInput, _b.Scale(Chunk(0), 1f, 1f));
        }

        var attended = Norm(Attention(attentionInput, positions, keyBias, rows, p + "attention."), p + "attention_norm2");
        x = _b.Add(x, mod is null ? attended : _b.Mul(_b.Unary("Tanh", Chunk(1)), attended));
        var ffnInput = Norm(x, p + "ffn_norm1");
        if (mod is not null)
        {
            ffnInput = _b.Mul(ffnInput, _b.Scale(Chunk(2), 1f, 1f));
        }

        var ffn = Norm(FeedForward(ffnInput, p + "ffn."), p + "ffn_norm2");
        return _b.Add(x, mod is null ? ffn : _b.Mul(_b.Unary("Tanh", Chunk(3)), ffn));
    }

    private Value Norm(Value x, string module) => _b.RmsNorm(x, _b.Weight(module + ".weight"), _hp.NormEps);

    /// <summary>Self-attention with per-head RMS-normed queries and keys and 3-axis RoPE; returns rows [T, D].</summary>
    private Value Attention(Value x, Value positions, Value? keyBias, int rows, string p)
    {
        int d = _hp.HiddenSize;
        var qkv = _b.Linear(x, p + "qkv.weight");
        Value Heads(long offset) => _b.Reshape(_b.Contiguous(_b.Slice(qkv, 1, offset, d)), rows, _hp.HeadCount, _hp.HeadDim);
        var q = _b.RmsNorm(Heads(0), _b.Weight(p + "norm_q.weight"), 1e-5f);
        var k = _b.RmsNorm(Heads(d), _b.Weight(p + "norm_k.weight"), 1e-5f);
        var v = Heads(2L * d);
        var rope = Attributes.From(("dims", _hp.HeadDim), ("base", _hp.RopeTheta), ("style", "gptj"),
            ("sections", new[] { _hp.AxesDim[0] / 2, _hp.AxesDim[1] / 2, _hp.AxesDim[2] / 2, 0 }));
        q = _b.Emit1("Rope", [q, positions, _frequencyFactors], rope);
        k = _b.Emit1("Rope", [k, positions, _frequencyFactors], rope);
        // f16 products with f32 accumulation: queries and keys are RMS-normed and Z-Image runs in bf16.
        var attended = _b.Emit1("Attention", keyBias is null ? [q, k, v] : [q, k, v, keyBias], Attributes.From(("scale", 1f / MathF.Sqrt(_hp.HeadDim)),
            ("causal", false), (StandardOps.F16ProductsAttribute, true)));
        return _b.Linear(_b.Reshape(attended, rows, d), p + "out.weight");
    }

    private Value FeedForward(Value x, string p)
        => _b.Linear(_b.Mul(_b.Silu(_b.Linear(x, p + "w1.weight")), _b.Linear(x, p + "w3.weight")), p + "w2.weight");

    /// <summary>Two linear layers with a SiLU between (the timestep embedder).</summary>
    private Value Mlp(Value x, string p) => Linear(_b.Silu(Linear(x, p + ".in_layer")), p + ".out_layer");

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", _b.HasWeight(module + ".bias") ? module + ".bias" : null);

    /// <summary>A row vector [1, D] repeated <paramref name="rows"/> times ([rows, D]).</summary>
    private Value Repeat(Value row, int rows, string name)
    {
        var ones = new float[rows];
        Array.Fill(ones, 1f);
        var column = _b.Constant(name + "_ones", DataType.F32, new Shape(rows, 1), MemoryMarshal.AsBytes(ones.AsSpan()).ToArray());
        return _b.Mul(column, _b.Emit1("Cast", [_b.Reshape(row, 1, _hp.HiddenSize)], Attributes.From(("type", "f32"))));
    }

    /// <summary>The sinusoidal embedding of values [1] into 256 channels, cosines first, frequencies 10000^(−i / 128).</summary>
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

    /// <summary>Latents [1, C, h, w] → patch rows [h/p · w/p, p·p·C], features ordered (row in patch, column in patch, channel).</summary>
    private Value Patchify(Value latent)
    {
        int p = _hp.PatchSize;
        int c = _hp.LatentChannels;
        long rows = _latentHeight / p;
        long columns = _latentWidth / p;
        var split = _b.Reshape(latent, c, rows, p, columns, p);
        var ordered = _b.Contiguous(_b.Permute(split, 1, 3, 2, 4, 0));
        return _b.Reshape(ordered, rows * columns, (long)p * p * c);
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
