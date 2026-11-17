using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="DiffusionGraphs.Denoise"/>) of a Qwen-Image 2.1 single-stream DiT for one latent size and one
/// text length: one token per latent pixel (no patching), the text features through a zero-centred RMS norm and a GELU MLP, then
/// blocks over the joint [text, image] sequence with block-causal attention (text rows attend causally to the text, image rows to all
/// visible text and the whole image), LayerNorm modulated by a scale and a tanh gate (no shift) from one shared modulation projection
/// (text rows from t = 0, image rows from the step's t), SwiGLU feed-forwards, and the scale-only final norm. RoPE over three axes
/// (16 + 56 + 56 of 128 head dims, θ 10000): text token j at (j, j, j), image tokens at (n, row − ⌈rows / 2⌉, column − ⌈columns / 2⌉)
/// with n the number of visible text tokens (computed in the graph from <see cref="DiffusionGraphs.ContextMask"/>). One image per run.
/// </summary>
/// <remarks>
/// Canonical weights (no biases): <c>transformer.img_in</c> [D, C], <c>transformer.txt_in.norm.weight</c> [context] (the effective scale,
/// stored + 1), <c>transformer.txt_in.in_layer|out_layer</c>, <c>transformer.time_in.in_layer|out_layer</c>, <c>transformer.modulation</c>
/// [4D, D] (scale and gate of the attention, then of the MLP), per block <c>transformer.blocks.{i}.attn.q|k|v|o</c>,
/// <c>attn.norm_q|norm_k.weight</c> [head dim], <c>mlp.gate|up</c> [M, D], <c>mlp.down</c> [D, M]; <c>transformer.final.mod</c> [D, D],
/// <c>transformer.final.linear</c> [C, D].
/// </remarks>
internal sealed class QwenImage21TransformerGraphBuilder
{
    private const string P = "transformer.";
    private const int FrequencyChannels = 256;
    private const float F16ProductBound = 50f;

    private readonly GraphBuilder _b;
    private readonly Func<string, WeightInfo?> _weights;
    private readonly QwenImage21HyperParameters _hp;
    private readonly int _textRows;
    private readonly int _imageRows;
    private readonly long _latentHeight;
    private readonly long _latentWidth;
    private Value _positions = null!;
    private Value _frequencyFactors = null!;
    private Value _imageKeyBias = null!;
    private Value _textSpans = null!;

    private QwenImage21TransformerGraphBuilder(Func<string, WeightInfo?> weights, QwenImage21HyperParameters hp, long latentHeight, long latentWidth, int textRows)
    {
        _b = new GraphBuilder(DiffusionGraphs.Denoise, weights);
        _weights = weights;
        _hp = hp;
        _textRows = textRows;
        _latentHeight = latentHeight;
        _latentWidth = latentWidth;
        _imageRows = (int)(latentHeight * latentWidth);
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    /// <param name="textRows">Rows of the text conditioning.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, QwenImage21HyperParameters hp, long latentHeight, long latentWidth, int textRows)
        => new QwenImage21TransformerGraphBuilder(weights, hp, latentHeight, latentWidth, textRows).Build();

    private ComputeGraph Build()
    {
        int d = _hp.HiddenSize;
        int c = _hp.LatentChannels;
        var latent = _b.Input(DiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(1, c, _latentHeight, _latentWidth));
        var timestep = _b.Input(DiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(1));
        var context = _b.Input(DiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(1, _textRows, _hp.ContextDim));
        var mask = _b.Input(DiffusionGraphs.ContextMask, InputRole.Other, DataType.F32, new Shape(1, _textRows));
        Value imageEmbedding;
        Value textEmbedding;
        using (_b.Group("conditioning"))
        {
            imageEmbedding = Mlp(Sinusoids(timestep, "time_frequencies"), P + "time_in");
            // t = 0: cos 0 = 1, sin 0 = 0.
            var zeroTime = new float[FrequencyChannels];
            zeroTime.AsSpan(0, FrequencyChannels / 2).Fill(1f);
            var zero = _b.Constant("zero_time", DataType.F32, new Shape(1, FrequencyChannels), MemoryMarshal.AsBytes(zeroTime.AsSpan()).ToArray());
            textEmbedding = _hp.CausalCondition ? Mlp(zero, P + "time_in") : imageEmbedding;
            DeclarePositions(mask);
        }

        var imageModulation = Linear(_b.Silu(imageEmbedding), P + "modulation");
        var textModulation = _hp.CausalCondition ? Linear(_b.Silu(textEmbedding), P + "modulation") : imageModulation;
        Value text;
        Value image;
        using (_b.Group("input"))
        {
            var features = _b.RmsNorm(_b.Reshape(context, _textRows, _hp.ContextDim), _b.Weight(P + "txt_in.norm.weight"), _hp.NormEps);
            text = Linear(_b.Unary("GeluTanh", Linear(features, P + "txt_in.in_layer")), P + "txt_in.out_layer");
            var rows = _b.Contiguous(_b.Permute(_b.Reshape(latent, c, _imageRows), 1, 0));
            image = Linear(rows, P + "img_in");
        }

        for (int i = 0; i < _hp.Layers; i++)
        {
            using var _ = _b.Group($"block.{i}");
            (text, image) = Block(text, image, textModulation, imageModulation, $"{P}blocks.{i}.");
        }

        using (_b.Group("output"))
        {
            var scale = Linear(_b.Silu(imageEmbedding), P + "final.mod");
            var normed = _b.Mul(LayerNorm(image), _b.Scale(scale, 1f, 1f));
            var rows = Linear(normed, P + "final.linear");
            _b.Output(DiffusionGraphs.Prediction, _b.Reshape(_b.Contiguous(_b.Permute(rows, 1, 0)), 1, c, _latentHeight, _latentWidth));
        }

        return _b.Build();
    }

    /// <summary>
    /// The RoPE positions i32 [text + image rows, 3] (text j at (j, j, j); image tokens at (n, centred row, centred column) with n the
    /// visible text tokens), the per-axis frequency divisors and the image rows' key bias over the text (masked padding −1e9).
    /// </summary>
    private void DeclarePositions(Value mask)
    {
        var visible = _b.Reshape(_b.Emit1("ReduceSum", [mask]), 1, 1);
        int gridRows = (int)_latentHeight;
        int columns = (int)_latentWidth;
        int rowOffset = gridRows - gridRows / 2;
        int columnOffset = columns - columns / 2;
        var fixedPart = new float[_imageRows * 3];
        var indicator = new float[_imageRows * 3];
        for (int t = 0; t < _imageRows; t++)
        {
            fixedPart[t * 3 + 1] = t / columns - rowOffset;
            fixedPart[t * 3 + 2] = t % columns - columnOffset;
            indicator[t * 3] = 1f;
        }

        var baseValues = _b.Constant("image_positions", DataType.F32, new Shape(_imageRows, 3), MemoryMarshal.AsBytes(fixedPart.AsSpan()).ToArray());
        var axis0 = _b.Constant("image_position_axis0", DataType.F32, new Shape(_imageRows, 3), MemoryMarshal.AsBytes(indicator.AsSpan()).ToArray());
        var image = _b.Emit1("Cast", [_b.Add(baseValues, _b.Mul(axis0, visible))], Attributes.From(("type", "i32")));
        var textValues = new int[_textRows * 3];
        for (int j = 0; j < _textRows; j++)
        {
            textValues[j * 3] = textValues[j * 3 + 1] = textValues[j * 3 + 2] = j;
        }

        var text = _b.Constant("text_positions", DataType.I32, new Shape(_textRows, 3), MemoryMarshal.AsBytes(textValues.AsSpan()).ToArray());
        _positions = _b.Concat(0, text, image);
        _frequencyFactors = AxialRope.FrequencyFactors(_b, "rope_frequency_factors", _hp.HeadDim, _hp.AxesDim, _hp.RopeTheta);
        // Causal attention over the text takes row spans: every text row's span is the whole text.
        var spans = new int[_textRows * 2];
        for (int j = 0; j < _textRows; j++)
        {
            spans[j * 2 + 1] = _textRows;
        }

        _textSpans = _b.Constant("text_spans", DataType.I32, new Shape(_textRows, 2), MemoryMarshal.AsBytes(spans.AsSpan()).ToArray());
        var zeros = _b.Constant("image_key_bias", DataType.F32, new Shape(_imageRows), new byte[_imageRows * sizeof(float)]);
        _imageKeyBias = _b.Concat(0, _b.Reshape(_b.Scale(mask, 1e9f, -1e9f), _textRows), zeros);
    }

    /// <summary>One block: modulated attention and SwiGLU over both streams with shared weights, block-causal attention.</summary>
    private (Value Text, Value Image) Block(Value text, Value image, Value textModulation, Value imageModulation, string p)
    {
        int d = _hp.HiddenSize;
        int rows = _textRows + _imageRows;
        var x = _b.Concat(0, Modulate(text, Chunk(textModulation, 0, d)), Modulate(image, Chunk(imageModulation, 0, d)));
        var q = _b.RmsNorm(Heads(Linear(x, p + "attn.q"), rows), _b.Weight(p + "attn.norm_q.weight"), _hp.NormEps);
        var k = _b.RmsNorm(Heads(Linear(x, p + "attn.k"), rows), _b.Weight(p + "attn.norm_k.weight"), _hp.NormEps);
        var v = Heads(Linear(x, p + "attn.v"), rows);
        var rope = AxialRope.Attributes(_hp.HeadDim, _hp.AxesDim, _hp.RopeTheta);
        q = _b.Emit1("Rope", [q, _positions, _frequencyFactors], rope);
        k = _b.Emit1("Rope", [k, _positions, _frequencyFactors], rope);
        bool f16 = F16Safe(p + "attn.");
        var scale = 1f / MathF.Sqrt(_hp.HeadDim);
        // Text rows: causal over the text. Image rows: every key, the text padding masked.
        var textAttention = _b.Emit1("Attention", [Part(q, 0, _textRows), Part(k, 0, _textRows), Part(v, 0, _textRows), _textSpans],
            Attributes.From(("scale", scale), ("causal", true), (StandardOps.F16ProductsAttribute, f16)));
        var imageAttention = _b.Emit1("Attention", [Part(q, _textRows, _imageRows), k, v, _imageKeyBias],
            Attributes.From(("scale", scale), ("causal", false), (StandardOps.F16ProductsAttribute, f16)));
        var attended = Linear(_b.Concat(0, _b.Reshape(textAttention, _textRows, d), _b.Reshape(imageAttention, _imageRows, d)), p + "attn.o");
        text = _b.Add(text, _b.Mul(Gate(textModulation, 1), _b.Slice(attended, 0, 0, _textRows)));
        image = _b.Add(image, _b.Mul(Gate(imageModulation, 1), _b.Slice(attended, 0, _textRows, _imageRows)));
        var y = _b.Concat(0, Modulate(text, Chunk(textModulation, 2, d)), Modulate(image, Chunk(imageModulation, 2, d)));
        var mlp = FeedForward(y, p + "mlp");
        text = _b.Add(text, _b.Mul(Gate(textModulation, 3), _b.Slice(mlp, 0, 0, _textRows)));
        image = _b.Add(image, _b.Mul(Gate(imageModulation, 3), _b.Slice(mlp, 0, _textRows, _imageRows)));
        return (text, image);
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

    private Value Part(Value heads, long first, long rows) => _b.Contiguous(_b.Slice(heads, 0, first, rows));

    private Value Heads(Value x, int rows) => _b.Reshape(x, rows, _hp.HeadCount, _hp.HeadDim);

    /// <summary>LayerNorm without affine parameters, then x · (1 + scale).</summary>
    private Value Modulate(Value x, Value scale) => _b.Mul(LayerNorm(x), _b.Scale(scale, 1f, 1f));

    private Value LayerNorm(Value x) => _b.Emit1("LayerNorm", [x], Attributes.From(("eps", _hp.NormEps)));

    /// <summary>tanh of chunk <paramref name="index"/> of the modulation [1, 4D].</summary>
    private Value Gate(Value modulation, int index) => _b.Unary("Tanh", Chunk(modulation, index, _hp.HiddenSize));

    private Value Chunk(Value modulation, int index, int width) => _b.Slice(modulation, 1, (long)index * width, width);

    /// <summary>The SwiGLU feed-forward; the down projection keeps f32 activations in the integer-dot kernels of quantized weights.</summary>
    private Value FeedForward(Value x, string p)
    {
        var hidden = _b.Mul(_b.Silu(Linear(x, p + ".gate")), Linear(x, p + ".up"));
        return _b.Emit1("MatMul", [hidden, _b.Weight(p + ".down.weight")], Attributes.From((StandardOps.F32ActivationsAttribute, true)));
    }

    private Value Mlp(Value x, string p) => Linear(_b.Silu(Linear(x, p + ".in_layer")), p + ".out_layer");

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", _b.HasWeight(module + ".bias") ? module + ".bias" : null);

    /// <summary>The sinusoidal embedding of values [1] (the timestep σ · 1000) into 256 channels, cosines first, 10000^(−i / 128).</summary>
    private Value Sinusoids(Value values, string name)
    {
        int half = FrequencyChannels / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)Math.Exp(-Math.Log(10000) * i / half);
        }

        var table = _b.Constant(name, DataType.F32, new Shape(1, half), MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var arguments = _b.Mul(_b.Reshape(values, 1, 1), table);
        return _b.Concat(1, _b.Unary("Cos", arguments), _b.Unary("Sin", arguments));
    }
}
