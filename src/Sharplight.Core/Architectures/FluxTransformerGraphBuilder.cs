using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="DiffusionGraphs.Denoise"/>) of a FLUX-style rectified-flow transformer for one latent size
/// and one prompt length: the latents are cut into p×p patches (tokens), the timestep (and, guidance-distilled, the guidance scale)
/// and the pooled text embedding form the modulation vector, dual-stream blocks attend jointly over the text and image tokens with
/// separate weights per stream, single-stream blocks run over the joined sequence (attention and MLP in parallel), and the image
/// tokens are projected back to latents. Attention uses per-head RMS-normed queries and keys and 3-axis RoPE over (0, row, column)
/// positions (text tokens at the origin). One image per run (a fixed batch of 1).
/// </summary>
/// <remarks>
/// Canonical weights (each <c>.weight</c> with its <c>.bias</c>, except the norm scales): <c>transformer.img_in</c> [D, C·p²],
/// <c>transformer.txt_in</c> [D, context], <c>transformer.time_in.in_layer|out_layer</c>, <c>transformer.vector_in.in_layer|out_layer</c>,
/// optional <c>transformer.guidance_in.in_layer|out_layer</c>; per dual-stream block <c>transformer.double.{i}.</c>
/// <c>{img,txt}_mod</c> [6D, D] (shift, scale, gate of the attention, then of the MLP), <c>{img,txt}_attn.qkv</c> [3D, D],
/// <c>{img,txt}_attn.norm_q|norm_k.weight</c> [head dim], <c>{img,txt}_attn.proj</c>, <c>{img,txt}_mlp.up|down</c>; per single-stream
/// block <c>transformer.single.{i}.</c> <c>mod</c> [3D, D], <c>linear1</c> [3D + M, D] (q, k, v, MLP input), <c>norm_q|norm_k.weight</c>,
/// <c>linear2</c> [D, D + M]; <c>transformer.final.mod</c> [2D, D] (shift, scale), <c>transformer.final.linear</c> [C·p², D].
/// Patch features are ordered (channel, row in patch, column in patch).
/// </remarks>
internal sealed class FluxTransformerGraphBuilder
{
    private const string P = "transformer.";
    private const float NormEps = 1e-6f;
    private const int FrequencyChannels = 256;

    private readonly GraphBuilder _b;
    private readonly FluxHyperParameters _hp;
    private readonly int _textRows;
    private readonly int _imageRows;
    private readonly long _latentHeight;
    private readonly long _latentWidth;
    private Value _positions = null!;
    private Value _frequencyFactors = null!;
    private Value? _modulations;
    private Value? _keyBias;
    private Value _modulationInput = null!;

    private FluxTransformerGraphBuilder(Func<string, WeightInfo?> weights, FluxHyperParameters hp, long latentHeight, long latentWidth, int textRows)
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
    /// <param name="textRows">Tokens of the text conditioning.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, FluxHyperParameters hp, long latentHeight, long latentWidth, int textRows)
        => new FluxTransformerGraphBuilder(weights, hp, latentHeight, latentWidth, textRows).Build();

    private ComputeGraph Build()
    {
        int d = _hp.HiddenSize;
        var latent = _b.Input(DiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(1, _hp.LatentChannels, _latentHeight, _latentWidth));
        var timestep = _b.Input(DiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(1));
        var context = _b.Input(DiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(1, _textRows, _hp.ContextDim));
        DeclarePositions();
        if (_hp.MaskedText)
        {
            // Masked text keys get −1e9 (0 → −1e9, 1 → 0); the image keys none.
            var mask = _b.Input(DiffusionGraphs.ContextMask, InputRole.Other, DataType.F32, new Shape(1, _textRows));
            var zeros = _b.Constant("image_key_bias", DataType.F32, new Shape(_imageRows), new byte[_imageRows * sizeof(float)]);
            _keyBias = _b.Concat(0, _b.Reshape(_b.Scale(mask, 1e9f, -1e9f), _textRows), zeros);
        }

        using (_b.Group("conditioning"))
        {
            if (_hp.ApproximatorLayers > 0)
            {
                _modulations = Approximator(timestep);
            }
            else
            {
                var pooled = _b.Input(DiffusionGraphs.Pooled, InputRole.Other, DataType.F32, new Shape(1, _hp.PooledDim));
                var guidance = _hp.GuidanceEmbed ? _b.Input(DiffusionGraphs.Guidance, InputRole.Other, DataType.F32, new Shape(1)) : null;
                var vector = Mlp(Sinusoids(timestep, "time_frequencies", FrequencyChannels), P + "time_in");
                if (guidance is not null)
                {
                    vector = _b.Add(vector, Mlp(Sinusoids(_b.Scale(guidance, 1000f), "guidance_frequencies", FrequencyChannels), P + "guidance_in"));
                }

                vector = _b.Add(vector, Mlp(pooled, P + "vector_in"));
                _modulationInput = _b.Silu(vector);
            }
        }

        Value image;
        Value text;
        using (_b.Group("input"))
        {
            image = Linear(Patchify(latent), P + "img_in");
            text = Linear(_b.Reshape(context, _textRows, _hp.ContextDim), P + "txt_in");
        }

        for (int i = 0; i < _hp.DoubleLayers; i++)
        {
            using var _ = _b.Group($"double.{i}");
            (image, text) = DoubleBlock(image, text, i, $"{P}double.{i}.");
        }

        var x = _b.Concat(0, text, image);
        for (int i = 0; i < _hp.SingleLayers; i++)
        {
            using var _ = _b.Group($"single.{i}");
            x = SingleBlock(x, i, $"{P}single.{i}.");
        }

        using (_b.Group("output"))
        {
            image = _b.Slice(x, 0, _textRows, _imageRows);
            var final = Modulation(P + "final.mod", 2, ModulationRows - 2);
            var normed = Modulate(image, Chunk(final, 0, d), Chunk(final, 1, d));
            _b.Output(DiffusionGraphs.Prediction, Unpatchify(Linear(normed, P + "final.linear")));
        }

        return _b.Build();
    }

    /// <summary>
    /// The constant RoPE positions [text + image rows, 3] (text at the origin, image tokens at (0, row, column)) and the per-frequency
    /// divisors that turn the op's single geometric frequency series into FLUX's per-axis series θ^(−2j / axis dims).
    /// </summary>
    private void DeclarePositions()
    {
        int rows = _textRows + _imageRows;
        var positions = new int[rows * 3];
        long columns = _latentWidth / _hp.PatchSize;
        for (int t = 0; t < _imageRows; t++)
        {
            int row = _textRows + t;
            positions[row * 3 + 1] = (int)(t / columns);
            positions[row * 3 + 2] = (int)(t % columns);
        }

        _positions = _b.Constant("rope_positions", DataType.I32, new Shape(rows, 3), MemoryMarshal.AsBytes(positions.AsSpan()).ToArray());
        int headDim = _hp.HeadDim;
        var factors = new float[headDim / 2];
        int pair = 0;
        foreach (int axisDims in _hp.AxesDim)
        {
            for (int j = 0; j < axisDims / 2; j++, pair++)
            {
                double op = Math.Pow(_hp.RopeTheta, -2.0 * pair / headDim);
                double flux = Math.Pow(_hp.RopeTheta, -2.0 * j / axisDims);
                factors[pair] = (float)(op / flux);
            }
        }

        if (pair != headDim / 2)
        {
            throw new InvalidDataException($"FLUX RoPE axes {string.Join("+", _hp.AxesDim)} do not cover the head width {headDim}.");
        }

        _frequencyFactors = _b.Constant("rope_frequency_factors", DataType.F32, new Shape(headDim / 2), MemoryMarshal.AsBytes(factors.AsSpan()).ToArray());
    }

    /// <summary>A dual-stream block: modulated pre-norm joint attention and modulated MLPs, each stream with its own weights.</summary>
    private (Value Image, Value Text) DoubleBlock(Value image, Value text, int index, string p)
    {
        int d = _hp.HiddenSize;
        var imageMod = Modulation(p + "img_mod", 6, 3 * _hp.SingleLayers + 6 * index);
        var textMod = Modulation(p + "txt_mod", 6, 3 * _hp.SingleLayers + 6 * _hp.DoubleLayers + 6 * index);
        var (iq, ik, iv) = Projections(Modulate(image, Chunk(imageMod, 0, d), Chunk(imageMod, 1, d)), p + "img_attn", _imageRows);
        var (tq, tk, tv) = Projections(Modulate(text, Chunk(textMod, 0, d), Chunk(textMod, 1, d)), p + "txt_attn", _textRows);
        var attended = Attend(_b.Concat(0, tq, iq), _b.Concat(0, tk, ik), _b.Concat(0, tv, iv));
        var textAttention = _b.Slice(attended, 0, 0, _textRows);
        var imageAttention = _b.Slice(attended, 0, _textRows, _imageRows);
        image = _b.Add(image, _b.Mul(Chunk(imageMod, 2, d), Linear(imageAttention, p + "img_attn.proj")));
        image = _b.Add(image, _b.Mul(Chunk(imageMod, 5, d), FeedForward(Modulate(image, Chunk(imageMod, 3, d), Chunk(imageMod, 4, d)), p + "img_mlp")));
        text = _b.Add(text, _b.Mul(Chunk(textMod, 2, d), Linear(textAttention, p + "txt_attn.proj")));
        text = _b.Add(text, _b.Mul(Chunk(textMod, 5, d), FeedForward(Modulate(text, Chunk(textMod, 3, d), Chunk(textMod, 4, d)), p + "txt_mlp")));
        return (image, text);
    }

    /// <summary>A single-stream block: one projection to q, k, v and the MLP input, attention and MLP side by side, one output projection.</summary>
    private Value SingleBlock(Value x, int index, string p)
    {
        int d = _hp.HiddenSize;
        int rows = _textRows + _imageRows;
        var mod = Modulation(p + "mod", 3, 3 * index);
        var projected = Linear(Modulate(x, Chunk(mod, 0, d), Chunk(mod, 1, d)), p + "linear1");
        long mlpWidth = projected.Shape[1].Fixed - 3 * d;
        var q = _b.RmsNorm(Heads(_b.Contiguous(_b.Slice(projected, 1, 0, d)), rows), _b.Weight(p + "norm_q.weight"), NormEps);
        var k = _b.RmsNorm(Heads(_b.Contiguous(_b.Slice(projected, 1, d, d)), rows), _b.Weight(p + "norm_k.weight"), NormEps);
        var v = Heads(_b.Contiguous(_b.Slice(projected, 1, 2 * d, d)), rows);
        var mlp = _b.Unary("GeluTanh", _b.Slice(projected, 1, 3 * d, mlpWidth));
        var joined = _b.Concat(1, Attend(q, k, v), mlp);
        return _b.Add(x, _b.Mul(Chunk(mod, 2, d), Linear(joined, p + "linear2")));
    }

    /// <summary>The RMS-normed query, key and value heads of one stream.</summary>
    private (Value Q, Value K, Value V) Projections(Value x, string p, int rows)
    {
        int d = _hp.HiddenSize;
        var qkv = Linear(x, p + ".qkv");
        var q = _b.RmsNorm(Heads(_b.Contiguous(_b.Slice(qkv, 1, 0, d)), rows), _b.Weight(p + ".norm_q.weight"), NormEps);
        var k = _b.RmsNorm(Heads(_b.Contiguous(_b.Slice(qkv, 1, d, d)), rows), _b.Weight(p + ".norm_k.weight"), NormEps);
        var v = Heads(_b.Contiguous(_b.Slice(qkv, 1, 2 * d, d)), rows);
        return (q, k, v);
    }

    /// <summary>Bidirectional attention over the joint sequence with RoPE on queries and keys; returns rows [T, D].</summary>
    private Value Attend(Value q, Value k, Value v)
    {
        int rows = _textRows + _imageRows;
        var rope = Attributes.From(("dims", _hp.HeadDim), ("base", _hp.RopeTheta), ("style", "gptj"),
            ("sections", new[] { _hp.AxesDim[0] / 2, _hp.AxesDim[1] / 2, _hp.AxesDim[2] / 2, 0 }));
        q = _b.Emit1("Rope", [q, _positions, _frequencyFactors], rope);
        k = _b.Emit1("Rope", [k, _positions, _frequencyFactors], rope);
        // f16 products with f32 accumulation (tensor-core flash attention): FLUX is trained and run in bf16, and the exact f32
        // kernel for 128-wide heads is the scalar one (96 % of a 1024×1024 step).
        var attended = _b.Emit1("Attention", _keyBias is null ? [q, k, v] : [q, k, v, _keyBias], Attributes.From(("scale", 1f / MathF.Sqrt(_hp.HeadDim)),
            ("causal", false), (StandardOps.F16ProductsAttribute, true)));
        return _b.Reshape(attended, rows, _hp.HiddenSize);
    }

    /// <summary>The modulation rows of Chroma's approximator, or of FLUX's projections of the conditioning vector.</summary>
    private int ModulationRows => 3 * _hp.SingleLayers + 12 * _hp.DoubleLayers + 2;

    /// <summary>
    /// A modulation vector [1, chunks · D]: with the approximator its rows [first, first + chunks) (Chroma), else the projection
    /// <paramref name="module"/> of the SiLU'd conditioning vector (FLUX).
    /// </summary>
    private Value Modulation(string module, int chunks, int first)
        => _modulations is { } rows
            ? _b.Reshape(_b.Contiguous(_b.Slice(rows, 0, first, chunks)), 1, (long)chunks * _hp.HiddenSize)
            : Linear(_modulationInput, module);

    /// <summary>
    /// Chroma's modulation approximator: per modulation row the sinusoidal embeddings of the timestep (16 channels), of guidance 0
    /// (16) and of the row index · 1000 (32), through an input projection, residual RMS-normed SiLU MLPs and an output projection,
    /// giving [rows, D].
    /// </summary>
    private Value Approximator(Value timestep)
    {
        int rows = ModulationRows;
        int quarter = _hp.ApproximatorChannels / 4;
        var time = Sinusoids(timestep, "time_frequencies", quarter);
        var ones = new float[rows];
        Array.Fill(ones, 1f);
        var broadcast = _b.Constant("approximator_rows", DataType.F32, new Shape(rows, 1), MemoryMarshal.AsBytes(ones.AsSpan()).ToArray());
        var fixedPart = new float[rows * 3 * quarter];
        int half = quarter / 2;
        int indexHalf = quarter;
        for (int row = 0; row < rows; row++)
        {
            int o = row * 3 * quarter;
            // Guidance 0: cos 0 = 1, sin 0 = 0.
            fixedPart.AsSpan(o, half).Fill(1f);
            // The row index · 1000 in 2 · quarter channels, computed in f32 as diffusers' get_timestep_embedding does.
            for (int i = 0; i < indexHalf; i++)
            {
                float exponent = (float)(-Math.Log(10000)) * i / indexHalf;
                float argument = row * 1000f * (float)Math.Exp(exponent);
                fixedPart[o + quarter + i] = (float)Math.Cos(argument);
                fixedPart[o + quarter + indexHalf + i] = (float)Math.Sin(argument);
            }
        }

        var constant = _b.Constant("approximator_fixed", DataType.F32, new Shape(rows, 3 * quarter), MemoryMarshal.AsBytes(fixedPart.AsSpan()).ToArray());
        var x = Linear(_b.Concat(1, _b.Mul(broadcast, time), constant), P + "approximator.in_proj");
        for (int layer = 0; layer < _hp.ApproximatorLayers; layer++)
        {
            string p = $"{P}approximator.layers.{layer}";
            // torch.nn.RMSNorm without eps: the f32 machine epsilon.
            var normed = _b.RmsNorm(x, _b.Weight($"{P}approximator.norms.{layer}.weight"), 1.1920929e-07f);
            x = _b.Add(x, Mlp(normed, p));
        }

        return Linear(x, P + "approximator.out_proj");
    }

    private Value Heads(Value x, int rows) => _b.Reshape(x, rows, _hp.HeadCount, _hp.HeadDim);

    /// <summary>LayerNorm without affine parameters, then x · (1 + scale) + shift (per-feature vectors [1, D]).</summary>
    private Value Modulate(Value x, Value shift, Value scale)
    {
        var normed = _b.Emit1("LayerNorm", [x], Attributes.From(("eps", NormEps)));
        return _b.Add(_b.Mul(normed, _b.Scale(scale, 1f, 1f)), shift);
    }

    /// <summary>Chunk <paramref name="index"/> of width <paramref name="width"/> of a modulation vector [1, k·D].</summary>
    private Value Chunk(Value modulation, int index, int width) => _b.Slice(modulation, 1, (long)index * width, width);

    private Value FeedForward(Value x, string p) => Linear(_b.Unary("GeluTanh", Linear(x, p + ".up")), p + ".down");

    /// <summary>Two linear layers with a SiLU between (the timestep, guidance and pooled-text embedders).</summary>
    private Value Mlp(Value x, string p) => Linear(_b.Silu(Linear(x, p + ".in_layer")), p + ".out_layer");

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", _b.HasWeight(module + ".bias") ? module + ".bias" : null);

    /// <summary>
    /// The sinusoidal embedding of values [1] into <paramref name="channels"/> channels, cosines first, frequencies 10000^(−i / half)
    /// (diffusers' <c>Timesteps(channels, flip_sin_to_cos=True, downscale_freq_shift=0)</c>).
    /// </summary>
    private Value Sinusoids(Value values, string name, int channels)
    {
        int half = channels / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)Math.Exp(-Math.Log(10000) * i / half);
        }

        var table = _b.Constant(name, DataType.F32, new Shape(1, half), MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
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

    /// <summary>Patch rows [h/p · w/p, C·p·p] → latents [1, C, h, w] (the inverse of <see cref="Patchify"/>).</summary>
    private Value Unpatchify(Value patches)
    {
        int p = _hp.PatchSize;
        int c = _hp.LatentChannels;
        long rows = _latentHeight / p;
        long columns = _latentWidth / p;
        var split = _b.Reshape(patches, rows, columns, c, p, p);
        var ordered = _b.Contiguous(_b.Permute(split, 2, 0, 3, 1, 4));
        return _b.Reshape(ordered, 1, c, _latentHeight, _latentWidth);
    }
}
