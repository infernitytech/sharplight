using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.ImageGeneration;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures;

/// <summary>
/// Builds the denoiser graph (<see cref="VideoDiffusionGraphs.Denoise"/>) of a HunyuanVideo-style video transformer for one latent size
/// and one text length: the text states are refined by a small bidirectional transformer conditioned on the timestep and their masked
/// mean, the latent video is cut into 1×2×2 patches (tokens in (frame, row, column) order), and dual-stream blocks (each stream with its
/// own modulated pre-norm projections and MLP, one joint attention) are followed by single-stream blocks (attention and MLP side by side)
/// over the video tokens followed by the text tokens; queries and keys are RMS-normed per head and the video tokens rotated by RoPE over
/// (frame, row, column); padded text keys are masked. The conditioning vector is the sum of the timestep, the embedded guidance scale and
/// the projected pooled CLIP embedding.
/// </summary>
/// <remarks>
/// Canonical weights: diffusers' <c>HunyuanVideoTransformer3DModel</c> names under <c>transformer.</c> (<c>x_embedder.proj</c> with the kernel
/// flattened to [D, C·pt·p·p], <c>context_embedder.*</c>, <c>time_text_embed.*</c>, <c>transformer_blocks.{i}.*</c>,
/// <c>single_transformer_blocks.{i}.*</c>, <c>norm_out.linear</c>, <c>proj_out</c>).
/// </remarks>
internal sealed class HunyuanVideoTransformerGraphBuilder
{
    private const string P = "transformer.";
    private const float NormEps = 1e-6f;
    private const int FrequencyChannels = 256;

    private readonly GraphBuilder _b;
    private readonly HunyuanVideoHyperParameters _hp;
    private readonly long _frames;
    private readonly long _rows;
    private readonly long _columns;
    private readonly int _textRows;
    private readonly int _videoRows;
    private Value _positions = null!;
    private Value _frequencyFactors = null!;
    private Value _keyBias = null!;
    private Value _textKeyBias = null!;
    private Value _modulationInput = null!;

    private HunyuanVideoTransformerGraphBuilder(Func<string, WeightInfo?> weights, HunyuanVideoHyperParameters hp, long latentFrames, long latentHeight, long latentWidth,
        int textRows)
    {
        if (hp.PatchSizeT != 1)
        {
            throw new NotSupportedException($"HunyuanVideo transformer: temporal patch size {hp.PatchSizeT} is not supported (only 1).");
        }

        _b = new GraphBuilder(VideoDiffusionGraphs.Denoise, weights);
        _hp = hp;
        _frames = latentFrames;
        _rows = latentHeight / hp.PatchSize;
        _columns = latentWidth / hp.PatchSize;
        _textRows = textRows;
        _videoRows = (int)(_frames * _rows * _columns);
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="latentFrames">Latent frames.</param>
    /// <param name="latentHeight">Latent height (a multiple of the patch size).</param>
    /// <param name="latentWidth">Latent width (a multiple of the patch size).</param>
    /// <param name="textRows">Tokens of the text conditioning.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, HunyuanVideoHyperParameters hp, long latentFrames, long latentHeight, long latentWidth, int textRows)
        => new HunyuanVideoTransformerGraphBuilder(weights, hp, latentFrames, latentHeight, latentWidth, textRows).Build();

    private ComputeGraph Build()
    {
        int d = _hp.HiddenSize;
        var latent = _b.Input(VideoDiffusionGraphs.Latent, InputRole.Other, DataType.F32, new Shape(_hp.InChannels, _frames, _rows * _hp.PatchSize, _columns * _hp.PatchSize));
        var timestep = _b.Input(VideoDiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(1));
        var context = _b.Input(VideoDiffusionGraphs.Context, InputRole.Other, DataType.F32, new Shape(_textRows, _hp.TextDim));
        var mask = _b.Input(VideoDiffusionGraphs.ContextMask, InputRole.Other, DataType.F32, new Shape(_textRows));
        var pooled = _b.Input(DiffusionGraphs.Pooled, InputRole.Other, DataType.F32, new Shape(1, _hp.PooledDim));
        var guidance = _hp.GuidanceEmbed ? _b.Input(DiffusionGraphs.Guidance, InputRole.Other, DataType.F32, new Shape(1)) : null;
        DeclarePositions();
        // (mask − 1) · 10⁹: padded text keys are ignored, the video keys never.
        _textKeyBias = _b.Reshape(_b.Scale(mask, 1e9f, -1e9f), 1, 1, _textRows);
        var zeros = _b.Constant("video_key_bias", DataType.F32, new Shape(_videoRows), new byte[_videoRows * sizeof(float)]);
        _keyBias = _b.Reshape(_b.Concat(0, zeros, _b.Reshape(_b.Scale(mask, 1e9f, -1e9f), _textRows)), 1, 1, _videoRows + _textRows);

        Value conditioning;
        Value text;
        using (_b.Group("conditioning"))
        {
            var time = Sinusoids(timestep, "time_frequencies");
            conditioning = Embedder(time, P + "time_text_embed.timestep_embedder");
            if (guidance is not null)
            {
                // The input is the guidance scale; the model embeds scale · 1000 like a timestep.
                conditioning = _b.Add(conditioning, Embedder(Sinusoids(_b.Scale(guidance, 1000f), "guidance_frequencies"), P + "time_text_embed.guidance_embedder"));
            }

            conditioning = _b.Add(conditioning, Embedder(pooled, P + "time_text_embed.text_embedder"));
            text = Refine(context, mask, time);
        }

        _modulationInput = _b.Silu(conditioning);
        Value video;
        using (_b.Group("input"))
        {
            video = Linear(Patchify(latent), P + "x_embedder.proj");
        }

        for (int i = 0; i < _hp.DoubleLayers; i++)
        {
            using var _ = _b.Group($"double.{i}");
            (video, text) = DoubleBlock(video, text, $"{P}transformer_blocks.{i}.");
        }

        var x = _b.Concat(0, video, text);
        for (int i = 0; i < _hp.SingleLayers; i++)
        {
            using var _ = _b.Group($"single.{i}");
            x = SingleBlock(x, $"{P}single_transformer_blocks.{i}.");
        }

        using (_b.Group("output"))
        {
            video = _b.Slice(x, 0, 0, _videoRows);
            var final = Linear(_modulationInput, P + "norm_out.linear");
            var normed = Modulate(video, Chunk(final, 1, d), Chunk(final, 0, d));
            _b.Output(VideoDiffusionGraphs.Prediction, Unpatchify(Linear(normed, P + "proj_out")));
        }

        return _b.Build();
    }

    /// <summary>
    /// The token refiner: the text states projected to the model width and refined by bidirectional blocks (affine layer norms, attention
    /// over the prompt's tokens, a SiLU MLP), whose residuals are gated by the timestep plus the masked mean of the raw text states.
    /// </summary>
    private Value Refine(Value context, Value mask, Value time)
    {
        string p = P + "context_embedder.";
        int d = _hp.HiddenSize;
        var maskColumn = _b.Reshape(mask, _textRows, 1);
        var sum = _b.Emit1("Reduce", [_b.Mul(context, maskColumn)], Attributes.From(("mode", "sum"), ("axes", new[] { 0L }), ("keepdims", true)));
        var count = _b.Emit1("Reduce", [maskColumn], Attributes.From(("mode", "sum"), ("axes", new[] { 0L }), ("keepdims", true)));
        var mean = _b.Mul(sum, _b.Unary("Recip", count));
        var vector = _b.Add(Embedder(time, p + "time_text_embed.timestep_embedder"), Embedder(mean, p + "time_text_embed.text_embedder"));
        var gates = _b.Silu(vector);
        var x = Linear(context, p + "proj_in");
        for (int i = 0; i < _hp.RefinerLayers; i++)
        {
            string r = $"{p}token_refiner.refiner_blocks.{i}.";
            var gate = Linear(gates, r + "norm_out.linear");
            var normed = DiffusionBlocks.LayerNorm(_b, x, r + "norm1", NormEps);
            var q = Heads(Linear(normed, r + "attn.to_q"), _textRows);
            var k = Heads(Linear(normed, r + "attn.to_k"), _textRows);
            var v = Heads(Linear(normed, r + "attn.to_v"), _textRows);
            var attended = _b.Reshape(_b.Emit1("Attention", [q, k, v, _textKeyBias], Attributes.From(("scale", 1f / MathF.Sqrt(_hp.HeadDim)), ("causal", false))),
                _textRows, d);
            x = _b.Add(x, _b.Mul(Linear(attended, r + "attn.to_out.0"), Chunk(gate, 0, d)));
            var hidden = Linear(_b.Silu(Linear(DiffusionBlocks.LayerNorm(_b, x, r + "norm2", NormEps), r + "ff.net.0.proj")), r + "ff.net.2");
            x = _b.Add(x, _b.Mul(hidden, Chunk(gate, 1, d)));
        }

        return x;
    }

    /// <summary>The constant RoPE positions [video + text rows, 3]: video tokens at (frame, row, column), text tokens at the origin (no rotation).</summary>
    private void DeclarePositions()
    {
        int rows = _videoRows + _textRows;
        var positions = new int[rows * 3];
        long perFrame = _rows * _columns;
        for (int t = 0; t < _videoRows; t++)
        {
            positions[t * 3] = (int)(t / perFrame);
            positions[t * 3 + 1] = (int)(t % perFrame / _columns);
            positions[t * 3 + 2] = (int)(t % _columns);
        }

        _positions = _b.Constant("rope_positions", DataType.I32, new Shape(rows, 3), MemoryMarshal.AsBytes(positions.AsSpan()).ToArray());
        _frequencyFactors = AxialRope.FrequencyFactors(_b, "rope_frequency_factors", _hp.HeadDim, _hp.AxesDim, _hp.RopeTheta);
    }

    /// <summary>A dual-stream block: each stream's modulated pre-norm projections and MLP, one joint attention.</summary>
    private (Value Video, Value Text) DoubleBlock(Value video, Value text, string p)
    {
        int d = _hp.HiddenSize;
        var videoMod = Linear(_modulationInput, p + "norm1.linear");
        var textMod = Linear(_modulationInput, p + "norm1_context.linear");
        var videoIn = Modulate(video, Chunk(videoMod, 0, d), Chunk(videoMod, 1, d));
        var textIn = Modulate(text, Chunk(textMod, 0, d), Chunk(textMod, 1, d));
        var (vq, vk, vv) = Projections(videoIn, p + "attn.to_q", p + "attn.to_k", p + "attn.to_v", p + "attn.norm_q", p + "attn.norm_k", _videoRows);
        var (tq, tk, tv) = Projections(textIn, p + "attn.add_q_proj", p + "attn.add_k_proj", p + "attn.add_v_proj", p + "attn.norm_added_q", p + "attn.norm_added_k", _textRows);
        var attended = Attend(_b.Concat(0, vq, tq), _b.Concat(0, vk, tk), _b.Concat(0, vv, tv));
        var videoAttention = _b.Slice(attended, 0, 0, _videoRows);
        var textAttention = _b.Slice(attended, 0, _videoRows, _textRows);
        video = _b.Add(video, _b.Mul(Linear(videoAttention, p + "attn.to_out.0"), Chunk(videoMod, 2, d)));
        text = _b.Add(text, _b.Mul(Linear(textAttention, p + "attn.to_add_out"), Chunk(textMod, 2, d)));
        video = _b.Add(video, _b.Mul(FeedForward(Modulate(video, Chunk(videoMod, 3, d), Chunk(videoMod, 4, d)), p + "ff"), Chunk(videoMod, 5, d)));
        text = _b.Add(text, _b.Mul(FeedForward(Modulate(text, Chunk(textMod, 3, d), Chunk(textMod, 4, d)), p + "ff_context"), Chunk(textMod, 5, d)));
        return (video, text);
    }

    /// <summary>A single-stream block: attention and a GELU MLP over the same modulated input, one output projection of both.</summary>
    private Value SingleBlock(Value x, string p)
    {
        int d = _hp.HiddenSize;
        int rows = _videoRows + _textRows;
        var mod = Linear(_modulationInput, p + "norm.linear");
        var normed = Modulate(x, Chunk(mod, 0, d), Chunk(mod, 1, d));
        var (q, k, v) = Projections(normed, p + "attn.to_q", p + "attn.to_k", p + "attn.to_v", p + "attn.norm_q", p + "attn.norm_k", rows);
        var mlp = _b.Unary("GeluTanh", Linear(normed, p + "proj_mlp"));
        var joined = _b.Concat(1, Attend(q, k, v), mlp);
        return _b.Add(x, _b.Mul(Linear(joined, p + "proj_out"), Chunk(mod, 2, d)));
    }

    /// <summary>Query, key and value heads with per-head RMS-normed queries and keys.</summary>
    private (Value Q, Value K, Value V) Projections(Value x, string q, string k, string v, string normQ, string normK, int rows)
    {
        var query = _b.RmsNorm(Heads(Linear(x, q), rows), _b.Weight(normQ + ".weight"), NormEps);
        var key = _b.RmsNorm(Heads(Linear(x, k), rows), _b.Weight(normK + ".weight"), NormEps);
        return (query, key, Heads(Linear(x, v), rows));
    }

    /// <summary>Joint attention (video rows, then text rows) with RoPE on the video rows and the padded text keys masked; returns rows [N, D].</summary>
    private Value Attend(Value q, Value k, Value v)
    {
        int rows = _videoRows + _textRows;
        var rope = AxialRope.Attributes(_hp.HeadDim, _hp.AxesDim, _hp.RopeTheta);
        q = _b.Emit1("Rope", [q, _positions, _frequencyFactors], rope);
        k = _b.Emit1("Rope", [k, _positions, _frequencyFactors], rope);
        // f16 products with f32 accumulation (tensor-core flash attention): HunyuanVideo is trained and run in bf16.
        var attended = _b.Emit1("Attention", [q, k, v, _keyBias], Attributes.From(("scale", 1f / MathF.Sqrt(_hp.HeadDim)), ("causal", false),
            (StandardOps.F16ProductsAttribute, true)));
        return _b.Reshape(attended, rows, _hp.HiddenSize);
    }

    private Value Heads(Value x, int rows) => _b.Reshape(_b.Contiguous(x), rows, _hp.HeadCount, _hp.HeadDim);

    /// <summary>LayerNorm without affine parameters, then x · (1 + scale) + shift (vectors [1, D]).</summary>
    private Value Modulate(Value x, Value shift, Value scale)
    {
        var normed = _b.Emit1("LayerNorm", [x], Attributes.From(("eps", NormEps)));
        return _b.Add(_b.Mul(normed, _b.Scale(scale, 1f, 1f)), shift);
    }

    private Value Chunk(Value modulation, int index, int width) => _b.Slice(modulation, 1, (long)index * width, width);

    private Value FeedForward(Value x, string p) => Linear(_b.Unary("GeluTanh", Linear(x, p + ".net.0.proj")), p + ".net.2");

    /// <summary>Two linear layers with a SiLU between (diffusers' <c>TimestepEmbedding</c> and <c>PixArtAlphaTextProjection</c>).</summary>
    private Value Embedder(Value x, string p) => Linear(_b.Silu(Linear(x, p + ".linear_1")), p + ".linear_2");

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", _b.HasWeight(module + ".bias") ? module + ".bias" : null);

    /// <summary>The sinusoidal embedding of values [1] into 256 channels, cosines first (diffusers' <c>Timesteps(256, flip_sin_to_cos=True, 0)</c>).</summary>
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

    /// <summary>Latents [C, T, h, w] → patch rows [T · h/p · w/p, C·p·p], features ordered (channel, row in patch, column in patch).</summary>
    private Value Patchify(Value latent)
    {
        int p = _hp.PatchSize;
        int c = _hp.InChannels;
        var split = _b.Reshape(latent, c, _frames, _rows, p, _columns, p);
        var ordered = _b.Contiguous(_b.Permute(split, 1, 2, 4, 0, 3, 5));
        return _b.Reshape(ordered, _videoRows, (long)c * p * p);
    }

    /// <summary>Output rows [tokens, C·p·p] with features ordered (channel, row in patch, column in patch) → latents [C, T, h, w].</summary>
    private Value Unpatchify(Value patches)
    {
        int p = _hp.PatchSize;
        int c = _hp.OutChannels;
        var split = _b.Reshape(patches, _frames, _rows, _columns, c, p, p);
        var ordered = _b.Contiguous(_b.Permute(split, 3, 0, 1, 4, 2, 5));
        return _b.Reshape(ordered, c, _frames, _rows * p, _columns * p);
    }
}
