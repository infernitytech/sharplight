using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="DiffusionGraphs.Denoise"/>) of a Krea 2 single-stream DiT for one latent size and one text length.
/// The text conditioning [L, layers · D_text] (the tapped text-encoder hidden states of every token) first goes through the text fusion:
/// blocks across the tapped layers of every token (batched attention over sequences of <c>layers</c> rows), a linear projector over the
/// layers, and refiner blocks across the tokens (the text padding masked as keys); then a zero-centred RMS norm and a tanh-GELU MLP into the
/// model width. The joint [text, image] sequence (image tokens: 2×2 latent patches) runs through blocks with RMS norms modulated by
/// (1 + scale) and a shift and residual gates from one shared time modulation plus a learned table per block, grouped-query attention with
/// q / k RMS norms, RoPE and a sigmoid output gate (masked text padding), and SwiGLU feed-forwards; the final RMS norm is modulated by the
/// time embedding plus a table. RoPE over three axes (32 + 48 + 48 of 128 head dims, θ 1000, interleaved pairs): text rows at the origin, image
/// tokens at (0, row, column). Every norm is an RMS norm whose (zero-centred) scale the importer stored + 1. One image per run.
/// </summary>
/// <remarks>
/// Canonical weights: <c>transformer.img_in</c> [D, C·4] (+ bias), <c>transformer.time_in.in_layer|out_layer</c>, <c>transformer.time_mod</c>
/// [6D, D], <c>transformer.text_fusion.{layerwise|refiner}.{i}.{norm1, norm2}.weight</c>, <c>.attn.q|k|v|gate|o</c>, <c>.attn.norm_q|norm_k.weight</c>,
/// <c>.mlp.gate|up|down</c>, <c>transformer.text_fusion.projector.weight</c> [1, layers], <c>transformer.txt_in.norm.weight</c>,
/// <c>transformer.txt_in.in_layer|out_layer</c>, per block <c>transformer.blocks.{i}.mod</c> [6, D] and the text-fusion block's parts,
/// <c>transformer.final.mod</c> [2, D], <c>transformer.final.norm.weight</c>, <c>transformer.final.linear</c> [C·4, D].
/// </remarks>
internal sealed class Krea2TransformerGraphBuilder
{
    private const string P = "transformer.";
    private const float F16ProductBound = 50f;

    private readonly GraphBuilder _b;
    private readonly Func<string, WeightInfo?> _weights;
    private readonly Krea2HyperParameters _hp;
    private readonly int _textRows;
    private readonly int _imageRows;
    private readonly long _latentHeight;
    private readonly long _latentWidth;
    private Value _positions = null!;
    private Value _frequencyFactors = null!;
    private Value _keyBias = null!;
    private Value _textKeyBias = null!;

    private Krea2TransformerGraphBuilder(Func<string, WeightInfo?> weights, Krea2HyperParameters hp, long latentHeight, long latentWidth, int textRows)
    {
        _b = new GraphBuilder(DiffusionGraphs.Denoise, weights);
        _weights = weights;
        _hp = hp;
        _textRows = textRows;
        _latentHeight = latentHeight;
        _latentWidth = latentWidth;
        _imageRows = (int)(latentHeight / hp.PatchSize * (latentWidth / hp.PatchSize));
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    /// <param name="textRows">Rows of the text conditioning.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, Krea2HyperParameters hp, long latentHeight, long latentWidth, int textRows)
        => new Krea2TransformerGraphBuilder(weights, hp, latentHeight, latentWidth, textRows).Build();

    private ComputeGraph Build()
    {
        int d = _hp.HiddenSize;
        var latent = _b.Input(DiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(1, _hp.LatentChannels, _latentHeight, _latentWidth));
        var timestep = _b.Input(DiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(1));
        var context = _b.Input(DiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(1, _textRows, (long)_hp.TextLayers * _hp.TextDim));
        var mask = _b.Input(DiffusionGraphs.ContextMask, InputRole.Other, DataType.F32, new Shape(1, _textRows));
        Value embedding;
        Value modulation;
        using (_b.Group("conditioning"))
        {
            int frequencies = (int)(_weights(P + "time_in.in_layer.weight")?.Shape[1].Fixed ?? 256);
            embedding = Linear(_b.Unary("GeluTanh", Linear(Sinusoids(timestep, frequencies), P + "time_in.in_layer")), P + "time_in.out_layer");
            modulation = _b.Reshape(Linear(_b.Unary("GeluTanh", embedding), P + "time_mod"), 6, d);
            DeclarePositions(mask);
        }

        Value text;
        using (_b.Group("text_fusion"))
        {
            text = TextFusion(context);
        }

        Value x;
        using (_b.Group("input"))
        {
            var features = _b.RmsNorm(text, _b.Weight(P + "txt_in.norm.weight"), _hp.NormEps);
            text = Linear(_b.Unary("GeluTanh", Linear(features, P + "txt_in.in_layer")), P + "txt_in.out_layer");
            x = _b.Concat(0, text, Linear(Patchify(latent), P + "img_in"));
        }

        for (int i = 0; i < _hp.Layers; i++)
        {
            using var _ = _b.Group($"block.{i}");
            x = Block(x, modulation, $"{P}blocks.{i}.");
        }

        using (_b.Group("output"))
        {
            var image = _b.Slice(x, 0, _textRows, _imageRows);
            var table = _b.Add(_b.Weight(P + "final.mod"), embedding);
            var normed = _b.RmsNorm(image, _b.Weight(P + "final.norm.weight"), _hp.NormEps);
            var modulated = _b.Add(_b.Mul(normed, _b.Scale(_b.Slice(table, 0, 0, 1), 1f, 1f)), _b.Slice(table, 0, 1, 1));
            _b.Output(DiffusionGraphs.Prediction, Unpatchify(Linear(modulated, P + "final.linear")));
        }

        return _b.Build();
    }

    /// <summary>
    /// The text fusion: [L, layers · D_text] → blocks across the layers of every token → the projector over the layers → refiner blocks
    /// across the tokens → [L, D_text].
    /// </summary>
    private Value TextFusion(Value context)
    {
        int layers = _hp.TextLayers;
        int width = _hp.TextDim;
        var x = _b.Reshape(context, (long)_textRows * layers, width);
        for (int i = 0; i < _hp.LayerwiseBlocks; i++)
        {
            using var _ = _b.Group($"layerwise.{i}");
            x = TextBlock(x, _textRows * layers, $"{P}text_fusion.layerwise.{i}.", layerwise: true);
        }

        // Linear over the layer axis: [L, layers, D] → [L · D, layers] · projectorᵀ.
        var columns = _b.Contiguous(_b.Permute(_b.Reshape(x, _textRows, layers, width), 0, 2, 1));
        x = _b.Reshape(Linear(_b.Reshape(columns, (long)_textRows * width, layers), P + "text_fusion.projector"), _textRows, width);
        for (int i = 0; i < _hp.RefinerBlocks; i++)
        {
            using var _ = _b.Group($"refiner.{i}");
            x = TextBlock(x, _textRows, $"{P}text_fusion.refiner.{i}.", layerwise: false);
        }

        return x;
    }

    /// <summary>A pre-norm text-fusion block (no RoPE, no modulation); layer-wise blocks attend within each token's layers, refiners across the unmasked tokens.</summary>
    private Value TextBlock(Value x, int rows, string p, bool layerwise)
    {
        var normed = _b.RmsNorm(x, _b.Weight(p + "norm1.weight"), _hp.NormEps);
        var (q, k, v, gate) = Projections(normed, rows, p, _hp.TextHeads, _hp.TextKeyValueHeads, _hp.TextHeadDim);
        var scale = 1f / MathF.Sqrt(_hp.TextHeadDim);
        Value attended;
        if (layerwise)
        {
            int layers = _hp.TextLayers;
            attended = _b.Emit1("Attention", [q, k, v],
                Attributes.From(("scale", scale), ("causal", false), ("sequence_rows", layers), ("sequence_keys", layers), (StandardOps.F16ProductsAttribute, F16Safe(p + "attn."))));
        }
        else
        {
            attended = _b.Emit1("Attention", [q, k, v, _textKeyBias],
                Attributes.From(("scale", scale), ("causal", false), (StandardOps.F16ProductsAttribute, F16Safe(p + "attn."))));
        }

        x = _b.Add(x, Linear(_b.Mul(_b.Reshape(attended, rows, _hp.TextDim), _b.Sigmoid(gate)), p + "attn.o"));
        return _b.Add(x, FeedForward(_b.RmsNorm(x, _b.Weight(p + "norm2.weight"), _hp.NormEps), p + "mlp"));
    }

    /// <summary>One block over [text, image]: modulated gated attention and SwiGLU, each with a residual gate.</summary>
    private Value Block(Value x, Value modulation, string p)
    {
        int rows = _textRows + _imageRows;
        var table = _b.Add(modulation, _b.Weight(p + "mod"));
        var h = Modulate(_b.RmsNorm(x, _b.Weight(p + "norm1.weight"), _hp.NormEps), Chunk(table, 0), Chunk(table, 1));
        var (q, k, v, gate) = Projections(h, rows, p, _hp.HeadCount, _hp.KeyValueHeads, _hp.HeadDim);
        var rope = AxialRope.Attributes(_hp.HeadDim, _hp.AxesDim, _hp.RopeTheta);
        q = _b.Emit1("Rope", [q, _positions, _frequencyFactors], rope);
        k = _b.Emit1("Rope", [k, _positions, _frequencyFactors], rope);
        var attended = _b.Emit1("Attention", [q, k, v, _keyBias],
            Attributes.From(("scale", 1f / MathF.Sqrt(_hp.HeadDim)), ("causal", false), (StandardOps.F16ProductsAttribute, F16Safe(p + "attn."))));
        var output = Linear(_b.Mul(_b.Reshape(attended, rows, _hp.HiddenSize), _b.Sigmoid(gate)), p + "attn.o");
        x = _b.Add(x, _b.Mul(Chunk(table, 2), output));
        var y = Modulate(_b.RmsNorm(x, _b.Weight(p + "norm2.weight"), _hp.NormEps), Chunk(table, 3), Chunk(table, 4));
        return _b.Add(x, _b.Mul(Chunk(table, 5), FeedForward(y, p + "mlp")));
    }

    /// <summary>The q (RMS-normed per head), k (RMS-normed per head), v heads and the output gate's pre-activation.</summary>
    private (Value Q, Value K, Value V, Value Gate) Projections(Value x, int rows, string p, int heads, int keyValueHeads, int headDim)
    {
        var q = _b.RmsNorm(_b.Reshape(Linear(x, p + "attn.q"), rows, heads, headDim), _b.Weight(p + "attn.norm_q.weight"), _hp.NormEps);
        var k = _b.RmsNorm(_b.Reshape(Linear(x, p + "attn.k"), rows, keyValueHeads, headDim), _b.Weight(p + "attn.norm_k.weight"), _hp.NormEps);
        var v = _b.Reshape(Linear(x, p + "attn.v"), rows, keyValueHeads, headDim);
        return (q, k, v, Linear(x, p + "attn.gate"));
    }

    /// <summary>
    /// The RoPE positions i32 [text + image rows, 3] (text at the origin, image tokens at (0, row, column)), the per-axis frequency
    /// divisors, and the key biases (masked text padding −1e9) of the joint sequence and of the text alone.
    /// </summary>
    private void DeclarePositions(Value mask)
    {
        int columns = (int)(_latentWidth / _hp.PatchSize);
        var values = new int[(_textRows + _imageRows) * 3];
        for (int t = 0; t < _imageRows; t++)
        {
            int row = (_textRows + t) * 3;
            values[row + 1] = t / columns;
            values[row + 2] = t % columns;
        }

        _positions = _b.Constant("positions", DataType.I32, new Shape(_textRows + _imageRows, 3), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
        _frequencyFactors = AxialRope.FrequencyFactors(_b, "rope_frequency_factors", _hp.HeadDim, _hp.AxesDim, _hp.RopeTheta);
        _textKeyBias = _b.Reshape(_b.Scale(mask, 1e9f, -1e9f), _textRows);
        var zeros = _b.Constant("image_key_bias", DataType.F32, new Shape(_imageRows), new byte[_imageRows * sizeof(float)]);
        _keyBias = _b.Concat(0, _textKeyBias, zeros);
    }

    /// <summary>Whether the block's RMS norm scales bound the scores well enough for f16 products (see FLUX's builder).</summary>
    private bool F16Safe(string p)
    {
        var q = _weights(p + "norm_q.weight");
        var k = _weights(p + "norm_k.weight");
        if (q is null || k is null || !WeightConversions.IsFloat(q.Type) || !WeightConversions.IsFloat(k.Type))
        {
            return true;
        }

        return WeightConversions.ReadFloats(q).Max(MathF.Abs) * WeightConversions.ReadFloats(k).Max(MathF.Abs) <= F16ProductBound;
    }

    /// <summary>x · (1 + scale) + shift.</summary>
    private Value Modulate(Value x, Value scale, Value shift) => _b.Add(_b.Mul(x, _b.Scale(scale, 1f, 1f)), shift);

    /// <summary>Row <paramref name="index"/> [1, D] of a modulation table [6, D].</summary>
    private Value Chunk(Value table, int index) => _b.Slice(table, 0, index, 1);

    /// <summary>The SwiGLU feed-forward; the down projection keeps f32 activations in the integer-dot kernels of quantized weights.</summary>
    private Value FeedForward(Value x, string p)
    {
        var hidden = _b.Mul(_b.Silu(Linear(x, p + ".gate")), Linear(x, p + ".up"));
        return _b.Emit1("MatMul", [hidden, _b.Weight(p + ".down.weight")], Attributes.From((StandardOps.F32ActivationsAttribute, true)));
    }

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", _b.HasWeight(module + ".bias") ? module + ".bias" : null);

    /// <summary>Latents [1, C, h, w] → patch rows [h/2 · w/2, C·4], features ordered (channel, row in patch, column in patch).</summary>
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

    /// <summary>Patch rows [h/2 · w/2, C·4] → latents [1, C, h, w] (the inverse of <see cref="Patchify"/>).</summary>
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

    /// <summary>The sinusoidal embedding of the timestep σ · 1000 into <paramref name="channels"/> channels, cosines first, 10000^(−i / half).</summary>
    private Value Sinusoids(Value values, int channels)
    {
        int half = channels / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)Math.Exp(-Math.Log(10000) * i / half);
        }

        var table = _b.Constant("time_frequencies", DataType.F32, new Shape(1, half), MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var arguments = _b.Mul(_b.Reshape(values, 1, 1), table);
        return _b.Concat(1, _b.Unary("Cos", arguments), _b.Unary("Sin", arguments));
    }
}
