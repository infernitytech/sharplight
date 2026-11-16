using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="VideoDiffusionGraphs.Denoise"/>) of an LTX-Video diffusion transformer for one latent size
/// and one prompt length: every latent pixel is a token (1×1×1 patches), the timestep gives a six-fold modulation of each block
/// (RMS-normed self-attention with RoPE, cross-attention to the projected T5 tokens with a key mask, a gated feed-forward) and the
/// output modulation. The RoPE of LTX rotates pairs of the whole model width by angles linear in fractional (frame, row, column)
/// coordinates (normalized by a base grid, the frame axis scaled by the VAE's temporal compression over the frame rate), with the
/// frequency index outer and the axis inner; the angles are a constant product of the coordinates and a frequency matrix.
/// </summary>
/// <remarks>
/// Canonical weights (diffusers' <c>LTXVideoTransformer3DModel</c> names under <c>transformer.</c>): <c>proj_in</c>,
/// <c>time_embed.emb.timestep_embedder.linear_1|linear_2</c>, <c>time_embed.linear</c> [6D, D], <c>caption_projection.linear_1|linear_2</c>;
/// per block <c>transformer_blocks.{i}.</c> <c>attn1|attn2.to_q|to_k|to_v|to_out.0</c>, <c>attn1|attn2.norm_q|norm_k.weight</c> [D],
/// <c>ff.net.0.proj</c>, <c>ff.net.2</c>, <c>scale_shift_table</c> f32 [6D]; <c>scale_shift_table</c> f32 [2D] and <c>proj_out</c>.
/// </remarks>
internal sealed class LtxTransformerGraphBuilder
{
    private const string P = "transformer.";
    private const float AttentionNormEps = 1e-5f;
    private const int FrequencyChannels = 256;

    private readonly GraphBuilder _b;
    private readonly LtxHyperParameters _hp;
    private readonly long _frames;
    private readonly long _rows;
    private readonly long _columns;
    private readonly int _textRows;
    private readonly int _tokens;
    private readonly float _framesPerSecond;
    private readonly int _temporalCompression;
    private readonly int _spatialCompression;
    private Value _cos = null!;
    private Value _sin = null!;

    private LtxTransformerGraphBuilder(Func<string, WeightInfo?> weights, LtxHyperParameters hp, long frames, long rows, long columns, int textRows,
        float framesPerSecond, int temporalCompression, int spatialCompression)
    {
        _b = new GraphBuilder(VideoDiffusionGraphs.Denoise, weights);
        _hp = hp;
        _frames = frames;
        _rows = rows;
        _columns = columns;
        _textRows = textRows;
        _tokens = (int)(frames * rows * columns);
        _framesPerSecond = framesPerSecond;
        _temporalCompression = temporalCompression;
        _spatialCompression = spatialCompression;
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="latentFrames">Latent frames.</param>
    /// <param name="latentHeight">Latent height.</param>
    /// <param name="latentWidth">Latent width.</param>
    /// <param name="textRows">Tokens of the text conditioning.</param>
    /// <param name="framesPerSecond">The frame rate the RoPE's frame axis is scaled for.</param>
    /// <param name="temporalCompression">Frames per latent frame of the VAE.</param>
    /// <param name="spatialCompression">Pixels per latent pixel of the VAE.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, LtxHyperParameters hp, long latentFrames, long latentHeight, long latentWidth, int textRows,
        float framesPerSecond, int temporalCompression, int spatialCompression)
        => new LtxTransformerGraphBuilder(weights, hp, latentFrames, latentHeight, latentWidth, textRows, framesPerSecond, temporalCompression, spatialCompression).Build();

    private ComputeGraph Build()
    {
        int d = _hp.HiddenSize;
        int c = _hp.LatentChannels;
        var latent = _b.Input(VideoDiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(c, _frames, _rows, _columns));
        var timestep = _b.Input(VideoDiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(1));
        var context = _b.Input(VideoDiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(_textRows, _hp.CaptionDim));
        var contextMask = _b.Input(VideoDiffusionGraphs.ContextMask, InputRole.Other, DataType.F32, new Shape(_textRows));
        DeclareRope();

        Value embedded;
        Value modulation;
        Value text;
        Value keyBias;
        using (_b.Group("conditioning"))
        {
            embedded = Linear(_b.Silu(Linear(Sinusoids(timestep), P + "time_embed.emb.timestep_embedder.linear_1")), P + "time_embed.emb.timestep_embedder.linear_2");
            modulation = Linear(_b.Silu(embedded), P + "time_embed.linear");
            text = Linear(_b.Unary("GeluTanh", Linear(context, P + "caption_projection.linear_1")), P + "caption_projection.linear_2");
            // (mask − 1) · 10⁴: the pipeline's additive key mask, broadcast over heads and query rows.
            keyBias = _b.Reshape(_b.Scale(contextMask, 10000f, -10000f), 1, 1, _textRows);
        }

        Value x;
        using (_b.Group("input"))
        {
            var packed = _b.Contiguous(_b.Permute(_b.Reshape(latent, c, _tokens), 1, 0));
            x = Linear(packed, P + "proj_in");
        }

        for (int i = 0; i < _hp.LayerCount; i++)
        {
            using var _ = _b.Group($"blocks.{i}");
            x = Block(x, text, keyBias, modulation, $"{P}transformer_blocks.{i}.");
        }

        using (_b.Group("output"))
        {
            var table = _b.Weight(P + "scale_shift_table");
            var shift = _b.Add(_b.Reshape(_b.Slice(table, 0, 0, d), 1, d), embedded);
            var scale = _b.Add(_b.Reshape(_b.Slice(table, 0, d, d), 1, d), embedded);
            var normed = _b.Emit1("LayerNorm", [x], Attributes.From(("eps", 1e-6f)));
            var projected = Linear(_b.Add(_b.Mul(normed, _b.Scale(scale, 1f, 1f)), shift), P + "proj_out");
            var unpacked = _b.Contiguous(_b.Permute(projected, 1, 0));
            _b.Output(VideoDiffusionGraphs.Prediction, _b.Reshape(unpacked, c, _frames, _rows, _columns));
        }

        return _b.Build();
    }

    /// <summary>
    /// The rotation tables cos / sin [tokens, D/2, 1]: angle (token t, pair p) = f_j · (2 c_a(t) − 1) with j = (p − pad) / 3,
    /// a = (p − pad) mod 3, f_j = θ^(j / (n − 1)) · π / 2 for n = D / 6 frequencies; the first pad = (D mod 6) / 2 pairs stay unrotated.
    /// </summary>
    private void DeclareRope()
    {
        int pairs = _hp.HiddenSize / 2;
        int count = _hp.HiddenSize / 6;
        int pad = pairs - 3 * count;
        var coordinates = new float[_tokens * 3];
        long perFrame = _rows * _columns;
        for (int t = 0; t < _tokens; t++)
        {
            long frame = t / perFrame;
            long row = t % perFrame / _columns;
            long column = t % _columns;
            coordinates[t * 3] = (float)(frame * (_temporalCompression / (double)_framesPerSecond) / _hp.RopeBase[0]);
            coordinates[t * 3 + 1] = (float)(row * (double)_spatialCompression / _hp.RopeBase[1]);
            coordinates[t * 3 + 2] = (float)(column * (double)_spatialCompression / _hp.RopeBase[2]);
        }

        var matrix = new float[3 * pairs];
        var offsets = new float[pairs];
        for (int j = 0; j < count; j++)
        {
            double exponent = count == 1 ? 0 : j / (double)(count - 1);
            float frequency = (float)(Math.Pow(_hp.RopeTheta, exponent) * Math.PI / 2);
            for (int a = 0; a < 3; a++)
            {
                int p = pad + j * 3 + a;
                matrix[p * 3 + a] = 2 * frequency;
                offsets[p] = -frequency;
            }
        }

        var coords = _b.Constant("rope_coordinates", DataType.F32, new Shape(_tokens, 3), MemoryMarshal.AsBytes(coordinates.AsSpan()).ToArray());
        // MatMul takes the matrix as [output, input] rows.
        var frequencies = _b.Constant("rope_frequencies", DataType.F32, new Shape(pairs, 3), MemoryMarshal.AsBytes(matrix.AsSpan()).ToArray());
        var bias = _b.Constant("rope_offsets", DataType.F32, new Shape(1, pairs), MemoryMarshal.AsBytes(offsets.AsSpan()).ToArray());
        var angles = _b.Add(_b.MatMul(coords, frequencies), bias);
        _cos = _b.Reshape(_b.Unary("Cos", angles), _tokens, pairs, 1);
        _sin = _b.Reshape(_b.Unary("Sin", angles), _tokens, pairs, 1);
    }

    /// <summary>Rotates the interleaved pairs of rows [tokens, D].</summary>
    private Value Rotate(Value x)
    {
        int pairs = _hp.HiddenSize / 2;
        var split = _b.Reshape(x, _tokens, pairs, 2);
        var real = _b.Slice(split, 2, 0, 1);
        var imaginary = _b.Slice(split, 2, 1, 1);
        var rotatedReal = _b.Sub(_b.Mul(real, _cos), _b.Mul(imaginary, _sin));
        var rotatedImaginary = _b.Add(_b.Mul(real, _sin), _b.Mul(imaginary, _cos));
        return _b.Reshape(_b.Concat(2, rotatedReal, rotatedImaginary), _tokens, _hp.HiddenSize);
    }

    private Value Block(Value x, Value text, Value keyBias, Value timeModulation, string p)
    {
        var mod = _b.Add(timeModulation, _b.Reshape(_b.Weight(p + "scale_shift_table"), 1, 6L * _hp.HiddenSize));
        var normed = _b.RmsNorm(x, null, _hp.NormEps);
        var attended = SelfAttention(_b.Add(_b.Mul(normed, _b.Scale(Chunk(mod, 1), 1f, 1f)), Chunk(mod, 0)), p + "attn1");
        x = _b.Add(x, _b.Mul(attended, Chunk(mod, 2)));
        x = _b.Add(x, CrossAttention(x, text, keyBias, p + "attn2"));
        var hidden = _b.Add(_b.Mul(_b.RmsNorm(x, null, _hp.NormEps), _b.Scale(Chunk(mod, 4), 1f, 1f)), Chunk(mod, 3));
        var feedForward = Linear(_b.Unary("GeluTanh", Linear(hidden, p + "ff.net.0.proj")), p + "ff.net.2");
        return _b.Add(x, _b.Mul(feedForward, Chunk(mod, 5)));
    }

    private Value SelfAttention(Value x, string p)
    {
        var q = Heads(Rotate(_b.RmsNorm(Linear(x, p + ".to_q"), _b.Weight(p + ".norm_q.weight"), AttentionNormEps)), _tokens);
        var k = Heads(Rotate(_b.RmsNorm(Linear(x, p + ".to_k"), _b.Weight(p + ".norm_k.weight"), AttentionNormEps)), _tokens);
        var v = Heads(Linear(x, p + ".to_v"), _tokens);
        var attended = _b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(_hp.HeadDim)), ("causal", false),
            (StandardOps.F16ProductsAttribute, true)));
        return Linear(_b.Reshape(attended, _tokens, _hp.HiddenSize), p + ".to_out.0");
    }

    /// <summary>Cross-attention with an additive key bias, written out as batched products over the heads.</summary>
    private Value CrossAttention(Value x, Value text, Value keyBias, string p)
    {
        Value HeadsFirst(Value rows, long count) => _b.Contiguous(_b.Permute(Heads(rows, count), 1, 0, 2));
        var q = HeadsFirst(_b.RmsNorm(Linear(x, p + ".to_q"), _b.Weight(p + ".norm_q.weight"), AttentionNormEps), _tokens);
        var k = HeadsFirst(_b.RmsNorm(Linear(text, p + ".to_k"), _b.Weight(p + ".norm_k.weight"), AttentionNormEps), _textRows);
        var v = HeadsFirst(Linear(text, p + ".to_v"), _textRows);
        var scores = _b.Add(_b.Scale(_b.Emit1("BatchMatMul", [q, k], Attributes.From(("trans_b", true))), 1f / MathF.Sqrt(_hp.HeadDim)), keyBias);
        var attended = _b.Emit1("BatchMatMul", [_b.Emit1("Softmax", [scores]), v]);
        var rows = _b.Reshape(_b.Contiguous(_b.Permute(attended, 1, 0, 2)), _tokens, _hp.HiddenSize);
        return Linear(rows, p + ".to_out.0");
    }

    private Value Heads(Value x, long rows) => _b.Reshape(_b.Contiguous(x), rows, _hp.HeadCount, _hp.HeadDim);

    private Value Chunk(Value modulation, int index) => _b.Slice(modulation, 1, (long)index * _hp.HiddenSize, _hp.HiddenSize);

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", _b.HasWeight(module + ".bias") ? module + ".bias" : null);

    /// <summary>The sinusoidal timestep embedding [1, 256], cosines first (diffusers' <c>Timesteps(256, True, 0)</c>).</summary>
    private Value Sinusoids(Value timesteps)
    {
        int half = FrequencyChannels / 2;
        var frequencies = new float[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = (float)Math.Exp(-Math.Log(10000) * i / half);
        }

        var table = _b.Constant("time_frequencies", DataType.F32, new Shape(1, half), MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var arguments = _b.Mul(_b.Reshape(timesteps, 1, 1), table);
        return _b.Concat(1, _b.Unary("Cos", arguments), _b.Unary("Sin", arguments));
    }
}
