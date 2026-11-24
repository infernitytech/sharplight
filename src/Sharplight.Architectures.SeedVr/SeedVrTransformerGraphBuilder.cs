using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;
using Sharplight.VideoGeneration;

namespace Sharplight.Architectures.SeedVr;

/// <summary>
/// Builds the restoration graph (<see cref="SeedVrGraphs.Restore"/>) of a SeedVR2 DiT for one latent size: the noisy latent, the encoded
/// input clip and a constant mask channel are cut into 1×2×2 patches (tokens in (frame, row, column) order) and projected; the precomputed
/// text embedding is projected to the model width; the timestep's sinusoidal embedding gives one table of shift, scale and gate rows that
/// every block adds its own learned rows to. Each block runs, per stream (video, text; the later blocks of the 3B share one set of weights),
/// an RMS norm without affine parameters, the modulation, a joint window attention (QKV projection, per-head RMS-normed queries and keys,
/// RoPE, each window of video tokens attending over itself and all text tokens, the text outputs averaged over the windows), the gated
/// output projection and residual, and the same around the feed-forward (SwiGLU or tanh-GELU). Even blocks use the plain window layout, odd
/// blocks the shifted one (<see cref="SeedVrWindowLayout"/>). The video stream is normed (3B: RMS norm with weight, then the modulation by
/// the table's first shift and scale rows plus the learned output rows) and projected back to patches.
/// </summary>
/// <remarks>
/// Canonical weights under <c>transformer.</c> (each <c>.weight</c> with its <c>.bias</c> where noted): <c>patch_in</c> [D, 4·C] (features in
/// (row in patch, column in patch, channel) order) and bias, <c>text_in</c> and bias, <c>time_embed.linear_1|2|3</c> and biases; per block
/// <c>blocks.{i}.{video|text|shared}.</c> <c>qkv.weight</c> [3·H·hd, D] (queries, keys, values, each head-major), <c>out</c> and bias,
/// <c>q_norm.weight</c> / <c>k_norm.weight</c> [hd], the feed-forward <c>mlp.gate|up|down.weight</c> (SwiGLU) or <c>mlp.up|down</c> with
/// biases (GELU), and the modulation rows <c>attn_shift|attn_scale|attn_gate|mlp_shift|mlp_scale|mlp_gate</c> [D]; <c>rope_frequencies</c>
/// [hd / 6 or fewer] (the per-axis frequency series), <c>norm_out.weight</c>, <c>out_shift</c> / <c>out_scale</c> (3B), <c>patch_out</c>
/// [4·C, D] and bias (features in (row in patch, column in patch, channel) order). The text embedding is <c>text_embedding.positive</c>.
/// </remarks>
internal sealed class SeedVrTransformerGraphBuilder
{
    private const string P = "transformer.";
    private const string TextEmbedding = "text_embedding.positive";

    private readonly GraphBuilder _b;
    private readonly SeedVrHyperParameters _hp;
    private readonly int _frames;
    private readonly int _rows;
    private readonly int _columns;
    private readonly int _tokens;
    private readonly int _textRows;
    private readonly WindowAttentionPlan[] _plans;
    private readonly WindowRotation[] _rotations;
    private Value _modulation = null!;

    private SeedVrTransformerGraphBuilder(Func<string, WeightInfo?> weights, SeedVrHyperParameters hp, int latentFrames, int latentHeight, int latentWidth)
    {
        if (hp.PatchSize[0] != 1)
        {
            throw new NotSupportedException($"SeedVR2 transformer: temporal patch size {hp.PatchSize[0]} is not supported (only 1).");
        }

        _b = new GraphBuilder(SeedVrGraphs.Restore, weights);
        _hp = hp;
        _frames = latentFrames;
        _rows = latentHeight / hp.PatchSize[1];
        _columns = latentWidth / hp.PatchSize[2];
        _tokens = _frames * _rows * _columns;
        var text = weights(TextEmbedding) ?? throw new InvalidDataException($"SeedVR2: the model has no '{TextEmbedding}'.");
        _textRows = (int)text.Shape[0].Fixed;
        var frequencies = ReadFloats(weights(P + "rope_frequencies") ?? throw new InvalidDataException("SeedVR2: the model has no 'transformer.rope_frequencies'."));
        _plans = new WindowAttentionPlan[2];
        _rotations = new WindowRotation[2];
        for (int layout = 0; layout < 2; layout++)
        {
            var windows = SeedVrWindowLayout.Windows(_frames, _rows, _columns, hp, shifted: layout == 1);
            _plans[layout] = WindowAttentionPlan.Create(windows, _frames, _rows, _columns, _textRows);
            _rotations[layout] = new WindowRotation(_b, hp, frequencies, windows, _rows, _columns, _textRows, $"layout{layout}");
        }
    }

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Canonical weight lookup.</param>
    /// <param name="hp">The transformer hyper-parameters.</param>
    /// <param name="latentFrames">Latent frames.</param>
    /// <param name="latentHeight">Latent height (a multiple of the patch height).</param>
    /// <param name="latentWidth">Latent width (a multiple of the patch width).</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, SeedVrHyperParameters hp, int latentFrames, int latentHeight, int latentWidth)
        => new SeedVrTransformerGraphBuilder(weights, hp, latentFrames, latentHeight, latentWidth).Build();

    private ComputeGraph Build()
    {
        long latentHeight = (long)_rows * _hp.PatchSize[1];
        long latentWidth = (long)_columns * _hp.PatchSize[2];
        var latentShape = new Shape(_hp.OutChannels, _frames, latentHeight, latentWidth);
        var latent = _b.Input(VideoDiffusionGraphs.Latent, InputRole.Other, DataType.F32, latentShape);
        var condition = _b.Input(SeedVrGraphs.Condition, InputRole.Other, DataType.F32, latentShape);
        var timestep = _b.Input(VideoDiffusionGraphs.Timestep, InputRole.Other, DataType.F32, new Shape(1));

        Value video;
        Value text;
        using (_b.Group("input"))
        {
            int maskChannels = _hp.InChannels - 2 * _hp.OutChannels;
            var mask = Ones("condition_mask", maskChannels * _frames * latentHeight * latentWidth);
            var stacked = _b.Concat(0, latent, condition, _b.Reshape(mask, maskChannels, _frames, latentHeight, latentWidth));
            video = Linear(Patchify(stacked, _hp.InChannels), P + "patch_in");
            text = Linear(_b.Weight(TextEmbedding), P + "text_in");
            var embedding = Linear(_b.Silu(Linear(_b.Silu(Linear(Sinusoids(timestep), P + "time_embed.linear_1")), P + "time_embed.linear_2")),
                P + "time_embed.linear_3");
            // [1, 6D] in (width, layer, shift|scale|gate) order → rows [6, D]: attention shift, scale, gate, then feed-forward shift, scale, gate.
            _modulation = _b.Contiguous(_b.Permute(_b.Reshape(embedding, _hp.Dim, 6), 1, 0));
        }

        for (int i = 0; i < _hp.Layers; i++)
        {
            using var _ = _b.Group($"blocks.{i}");
            (video, text) = Block(i, video, text);
        }

        using (_b.Group("output"))
        {
            if (_b.HasWeight(P + "norm_out.weight"))
            {
                video = _b.RmsNorm(video, _b.Weight(P + "norm_out.weight"), _hp.NormEps);
                // The official model reuses the attention layers' shift and scale rows of the table here (its modulation cache key).
                video = Modulate(video, Row(1, P + "out_scale"), Row(0, P + "out_shift"));
            }

            _b.Output(VideoDiffusionGraphs.Prediction, Unpatchify(Linear(video, P + "patch_out")));
        }

        return _b.Build();
    }

    /// <summary>One block over the video rows [tokens, D] and the text rows [L, D].</summary>
    private (Value Video, Value Text) Block(int layer, Value video, Value text)
    {
        string p = $"{P}blocks.{layer}.";
        string videoBranch = p + (_b.HasWeight(p + "video.qkv.weight") ? "video." : "shared.");
        string textBranch = p + (_b.HasWeight(p + "text.qkv.weight") ? "text." : "shared.");
        bool textPassthrough = _hp.LastLayerTextPassthrough && layer == _hp.Layers - 1;

        var videoIn = Modulate(_b.RmsNorm(video, null, _hp.NormEps), Row(1, videoBranch + "attn_scale"), Row(0, videoBranch + "attn_shift"));
        var textNormed = _b.RmsNorm(text, null, _hp.NormEps);
        var textIn = textPassthrough ? textNormed : Modulate(textNormed, Row(1, textBranch + "attn_scale"), Row(0, textBranch + "attn_shift"));
        var (videoAttended, textAttended) = Attention(videoIn, textIn, videoBranch, textBranch, layer % 2);

        video = _b.Add(video, _b.Mul(Linear(videoAttended, videoBranch + "out"), Row(2, videoBranch + "attn_gate")));
        var textOut = Linear(textAttended, textBranch + "out");
        text = _b.Add(text, textPassthrough ? textOut : _b.Mul(textOut, Row(2, textBranch + "attn_gate")));

        var videoHidden = Modulate(_b.RmsNorm(video, null, _hp.NormEps), Row(4, videoBranch + "mlp_scale"), Row(3, videoBranch + "mlp_shift"));
        video = _b.Add(video, _b.Mul(FeedForward(videoHidden, videoBranch + "mlp."), Row(5, videoBranch + "mlp_gate")));
        if (!textPassthrough)
        {
            var textHidden = Modulate(_b.RmsNorm(text, null, _hp.NormEps), Row(4, textBranch + "mlp_scale"), Row(3, textBranch + "mlp_shift"));
            text = _b.Add(text, _b.Mul(FeedForward(textHidden, textBranch + "mlp."), Row(5, textBranch + "mlp_gate")));
        }

        return (video, text);
    }

    /// <summary>The joint window attention: video rows [tokens, H·hd] and text rows [L, H·hd] before the output projections.</summary>
    private (Value Video, Value Text) Attention(Value video, Value text, string videoBranch, string textBranch, int layout)
    {
        int width = _hp.Heads * _hp.HeadDim;
        var videoQkv = Linear(video, videoBranch + "qkv");
        var textQkv = Linear(text, textBranch + "qkv");

        var rotation = _rotations[layout];
        // The official model runs its attention kernel in bf16 whatever the precision of the rest: queries, keys, values and the output
        // are rounded to bf16 (products accumulated in f32).
        var q = Bf16(Rows(rotation.RotateVideo(Normed(Part(videoQkv, 0), videoBranch + "q_norm")), rotation.RotateText(Normed(Part(textQkv, 0), textBranch + "q_norm"))));
        var k = Bf16(Rows(rotation.RotateVideo(Normed(Part(videoQkv, 1), videoBranch + "k_norm")), rotation.RotateText(Normed(Part(textQkv, 1), textBranch + "k_norm"))));
        var v = Bf16(Rows(Part(videoQkv, 2), Part(textQkv, 2)));

        var plan = _plans[layout];
        var outputs = new Value[plan.Groups.Count];
        for (int g = 0; g < plan.Groups.Count; g++)
        {
            var group = plan.Groups[g];
            var indices = Indices($"windows{layout}.{g}", group.Gather);
            outputs[g] = Bf16(_b.Reshape(Attend(GatherHeads(q, indices), GatherHeads(k, indices), GatherHeads(v, indices), group), group.Gather.Length, width));
        }

        var all = outputs.Length == 1 ? outputs[0] : _b.Concat(0, outputs);
        var videoOut = _b.Gather(all, Indices($"windows{layout}.tokens", plan.TokenRows));
        var textRows = _b.Reshape(_b.Gather(all, Indices($"windows{layout}.text", plan.TextRows)), plan.Windows.Count, _textRows, width);
        var textOut = _b.Emit1("Reduce", [textRows], Attributes.From(("mode", "mean"), ("axes", new long[] { 0 }), ("keepdims", false)));
        return (videoOut, textOut);
    }

    private Value Attend(Value q, Value k, Value v, WindowGroup group)
    {
        var attributes = new List<(string, object)>
        {
            ("scale", 1f / MathF.Sqrt(_hp.HeadDim)),
            ("causal", false),
            // f16 products with f32 accumulation: the official model attends in bf16.
            (StandardOps.F16ProductsAttribute, true),
        };
        if (group.Count > 1)
        {
            attributes.Add(("sequence_rows", group.Sequence));
            attributes.Add(("sequence_keys", group.Sequence));
        }

        return _b.Emit1("Attention", [q, k, v], Attributes.From([.. attributes]));
    }

    /// <summary>The values rounded to bf16 (and widened back to f32).</summary>
    private Value Bf16(Value x)
        => _b.Emit1("Cast", [_b.Emit1("Cast", [x], Attributes.From(("type", DataType.BF16.Name)))], Attributes.From(("type", DataType.F32.Name)));

    private Value GatherHeads(Value rows, Value indices)
    {
        var gathered = _b.Gather(rows, indices);
        return _b.Reshape(gathered, gathered.Shape[0].Fixed, _hp.Heads, _hp.HeadDim);
    }

    /// <summary>The video rows followed by the text rows, [tokens + L, H·hd].</summary>
    private Value Rows(Value video, Value text)
        => _b.Concat(0, _b.Reshape(video, _tokens, (long)_hp.Heads * _hp.HeadDim), _b.Reshape(text, _textRows, (long)_hp.Heads * _hp.HeadDim));

    /// <summary>Queries (0), keys (1) or values (2) of fused QKV rows, as heads [rows, H, hd].</summary>
    private Value Part(Value qkv, int index)
    {
        long width = (long)_hp.Heads * _hp.HeadDim;
        var part = _b.Contiguous(_b.Slice(qkv, 1, index * width, width));
        return _b.Reshape(part, qkv.Shape[0].Fixed, _hp.Heads, _hp.HeadDim);
    }

    private Value Normed(Value heads, string module) => _b.RmsNorm(heads, _b.Weight(module + ".weight"), _hp.NormEps);

    private Value FeedForward(Value x, string p)
    {
        if (_hp.GatedMlp)
        {
            return Linear(_b.Mul(_b.Silu(Linear(x, p + "gate")), Linear(x, p + "up")), p + "down");
        }

        return Linear(_b.Unary("GeluTanh", Linear(x, p + "up")), p + "down");
    }

    /// <summary>x · scale + shift with [1, D] rows.</summary>
    private Value Modulate(Value x, Value scale, Value shift) => _b.Add(_b.Mul(x, scale), shift);

    /// <summary>Row <paramref name="row"/> of the time-embedding table plus the learned row <paramref name="weight"/>, as [1, D].</summary>
    private Value Row(int row, string weight) => _b.Add(_b.Slice(_modulation, 0, row, 1), _b.Reshape(_b.Weight(weight), 1, _hp.Dim));

    private Value Linear(Value x, string module) => _b.Linear(x, module + ".weight", _b.HasWeight(module + ".bias") ? module + ".bias" : null);

    /// <summary>
    /// The sinusoidal embedding [1, F] of the timestep, sines first, frequencies 10000^(−i / (F/2)) (diffusers' <c>get_timestep_embedding</c>
    /// with <c>flip_sin_to_cos=False</c>, <c>downscale_freq_shift=0</c>).
    /// </summary>
    private Value Sinusoids(Value timestep)
    {
        int half = _hp.TimeEmbedDim / 2;
        var frequencies = new float[half];
        // In f32 as the reference computes them: at timestep 1000 an ulp of a frequency moves its angle noticeably.
        float logBase = (float)-Math.Log(10000);
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = MathF.Exp(logBase * i / half);
        }

        var table = _b.Constant("time_frequencies", DataType.F32, new Shape(1, half), MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var arguments = _b.Mul(_b.Reshape(timestep, 1, 1), table);
        return _b.Concat(1, _b.Unary("Sin", arguments), _b.Unary("Cos", arguments));
    }

    /// <summary>Latents [C, T, h, w] → patch rows [tokens, 2·2·C], features ordered (row in patch, column in patch, channel).</summary>
    private Value Patchify(Value latent, int channels)
    {
        int ph = _hp.PatchSize[1];
        int pw = _hp.PatchSize[2];
        var split = _b.Reshape(latent, channels, _frames, _rows, ph, _columns, pw);
        var ordered = _b.Contiguous(_b.Permute(split, 1, 2, 4, 3, 5, 0));
        return _b.Reshape(ordered, _tokens, (long)channels * ph * pw);
    }

    /// <summary>Output rows [tokens, 2·2·C] with features ordered (row in patch, column in patch, channel) → latents [C, T, h, w].</summary>
    private Value Unpatchify(Value patches)
    {
        int ph = _hp.PatchSize[1];
        int pw = _hp.PatchSize[2];
        int c = _hp.OutChannels;
        var split = _b.Reshape(patches, _frames, _rows, _columns, ph, pw, c);
        var ordered = _b.Contiguous(_b.Permute(split, 5, 0, 1, 3, 2, 4));
        return _b.Reshape(ordered, c, _frames, (long)_rows * ph, (long)_columns * pw);
    }

    private Value Indices(string name, int[] values)
        => _b.Constant(name, DataType.I32, new Shape(values.Length), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());

    private Value Ones(string name, long count)
    {
        var values = new float[count];
        Array.Fill(values, 1f);
        return _b.Constant(name, DataType.F32, new Shape(count), MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
    }

    private static float[] ReadFloats(WeightInfo weight)
    {
        if (weight.Type != DataType.F32)
        {
            throw new InvalidDataException($"SeedVR2: '{weight.Name}' must be f32.");
        }

        return MemoryMarshal.Cast<byte, float>(weight.ReadAll()).ToArray();
    }
}
