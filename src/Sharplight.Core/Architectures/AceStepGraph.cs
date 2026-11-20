using Sharplight.Audio;
using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Emits ACE-Step 1.5 as graphs of generic ops over time-major signals: the Qwen3-style caption encoder and lyric embedding, the condition encoder (lyric encoder, timbre
/// encoder and caption projection packed into one sequence), one evaluation of the diffusion transformer (patch embedding, adaptive-norm blocks of windowed self-attention,
/// cross-attention over the condition rows and a feed-forward layer, rotary positions, two timestep embeddings) and the autoencoder's decoder (<see cref="OobleckDecoderGraph"/>). Contract: <see cref="AudioGenerationGraphs"/>.
/// </summary>
internal static class AceStepGraph
{
    /// <summary>The caption encoder: token ids to the final hidden rows of a causal Qwen3 stack.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildText(AceStepParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioGenerationGraphs.TextGraph, weights);
        var rows = Dim.Sym(WellKnown.TokenSymbol);
        var tokens = b.Input(AudioGenerationGraphs.TokensInput, InputRole.TokenIds, DataType.I32, new Shape(rows));
        var positions = b.Input(AudioGenerationGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(rows, 1));
        var spans = b.Input(AudioGenerationGraphs.SpansInput, InputRole.RowSpans, DataType.I32, new Shape(rows, 2));
        var x = b.Gather(b.Weight("text.embed_tokens.weight"), tokens);
        var rope = Rope(p.TextHeadDim, p.TextRopeBase);
        var attention = Attention(p.TextHeadDim, causal: true, window: 0);
        for (int il = 0; il < p.TextLayers; il++)
        {
            using var _ = b.Group($"text.layers.{il}");
            x = Layer(b, x, $"text.layers.{il}.", positions, p.TextHeads, p.TextKvHeads, p.TextHeadDim, p.TextEps, rope, attention, spans);
        }

        b.Output(AudioGenerationGraphs.HiddenOutput, b.RmsNorm(x, b.Weight("text.norm.weight"), p.TextEps));
        return b.Build();
    }

    /// <summary>The lyric embedding: token ids to rows of the text encoder's embedding table.</summary>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildLyricEmbedding(Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioGenerationGraphs.LyricGraph, weights);
        var tokens = b.Input(AudioGenerationGraphs.TokensInput, InputRole.TokenIds, DataType.I32, new Shape(Dim.Sym(WellKnown.TokenSymbol)));
        b.Output(AudioGenerationGraphs.HiddenOutput, b.Gather(b.Weight("text.embed_tokens.weight"), tokens));
        return b.Build();
    }

    /// <summary>The condition encoder: lyric rows, timbre reference and caption rows packed in that order and projected to the denoiser's cross-attention input.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildCondition(AceStepParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioGenerationGraphs.ConditionGraph, weights);
        var lyricRows = Dim.Sym(AudioGenerationGraphs.LyricSymbol);
        var textRows = Dim.Sym(AudioGenerationGraphs.TextSymbol);
        var text = b.Input(AudioGenerationGraphs.TextInput, InputRole.Other, DataType.F32, new Shape(textRows, p.TextHidden));
        var lyric = b.Input(AudioGenerationGraphs.LyricInput, InputRole.Other, DataType.F32, new Shape(lyricRows, p.TextHidden));
        var reference = b.Input(AudioGenerationGraphs.ReferenceInput, InputRole.Other, DataType.F32, new Shape(p.ReferenceFrames, p.LatentChannels));
        var lyricPositions = b.Input(AudioGenerationGraphs.LyricPositionsInput, InputRole.Positions, DataType.I32, new Shape(lyricRows, 1));
        var referencePositions = b.Input(AudioGenerationGraphs.ReferencePositionsInput, InputRole.Positions, DataType.I32, new Shape(p.ReferenceFrames, 1));
        var rope = Rope(p.HeadDim, p.RopeBase);

        var lyricStates = Linear(b, lyric, "cond.lyric_encoder.embed_tokens");
        for (int il = 0; il < p.LyricLayers; il++)
        {
            using var _ = b.Group($"cond.lyric_encoder.layers.{il}");
            lyricStates = Layer(b, lyricStates, $"cond.lyric_encoder.layers.{il}.", lyricPositions, p.Heads, p.KvHeads, p.HeadDim, p.Eps, rope,
                Attention(p.HeadDim, causal: false, window: p.Sliding[il] ? p.AttentionWindow : 0));
        }

        lyricStates = b.RmsNorm(lyricStates, b.Weight("cond.lyric_encoder.norm.weight"), p.Eps);

        var timbre = Linear(b, reference, "cond.timbre_encoder.embed_tokens");
        for (int il = 0; il < p.TimbreLayers; il++)
        {
            using var _ = b.Group($"cond.timbre_encoder.layers.{il}");
            timbre = Layer(b, timbre, $"cond.timbre_encoder.layers.{il}.", referencePositions, p.Heads, p.KvHeads, p.HeadDim, p.Eps, rope,
                Attention(p.HeadDim, causal: false, window: p.Sliding[il] ? p.AttentionWindow : 0));
        }

        timbre = b.Slice(b.RmsNorm(timbre, b.Weight("cond.timbre_encoder.norm.weight"), p.Eps), 0, 0, 1);

        var projectedText = b.MatMul(text, b.Weight("cond.text_projector.weight"));
        var packed = b.Concat(0, lyricStates, timbre, projectedText);
        b.Output(AudioGenerationGraphs.ConditionOutput, Linear(b, packed, "dit.condition_embedder"));
        return b.Build();
    }

    /// <summary>One evaluation of the diffusion transformer.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDenoiser(AceStepParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioGenerationGraphs.DenoiseGraph, weights);
        var frames = Dim.Sym(WellKnown.TokenSymbol);
        var latent = b.Input(AudioGenerationGraphs.LatentInput, InputRole.Other, DataType.F32, new Shape(frames, p.LatentChannels));
        var context = b.Input(AudioGenerationGraphs.ContextInput, InputRole.Other, DataType.F32, new Shape(frames, p.InChannels - p.LatentChannels));
        var timestep = b.Input(AudioGenerationGraphs.TimestepInput, InputRole.Other, DataType.F32, new Shape(1));
        var timestepR = b.Input(AudioGenerationGraphs.TimestepRInput, InputRole.Other, DataType.F32, new Shape(1));
        var condition = b.Input(AudioGenerationGraphs.ConditionInput, InputRole.Other, DataType.F32, new Shape(Dim.Sym(WellKnown.EncoderSymbol), p.Hidden));
        var positions = b.Input(AudioGenerationGraphs.PositionsInput, InputRole.Positions, DataType.I32, new Shape(frames.FloorDiv(p.PatchSize), 1));

        var (temb, projection) = TimestepEmbedding(b, timestep, "dit.time_embed.", p.Hidden);
        var (tembR, projectionR) = TimestepEmbedding(b, b.Sub(timestep, timestepR), "dit.time_embed_r.", p.Hidden);
        temb = b.Add(temb, tembR);
        projection = b.Add(projection, projectionR);

        var x = b.Concat(1, context, latent);
        long patch = p.PatchSize;
        x = b.Emit1("Conv1d", [x, b.Weight("dit.proj_in.1.weight"), b.Weight("dit.proj_in.1.bias")],
            Attributes.From(("stride", patch), ("dilation", 1L), ("pad_left", 0L), ("pad_right", 0L)));
        var rope = Rope(p.HeadDim, p.RopeBase);
        for (int il = 0; il < p.Layers; il++)
        {
            using var _ = b.Group($"dit.layers.{il}");
            string n = $"dit.layers.{il}.";
            var table = b.Add(b.Weight(n + "scale_shift_table"), projection);
            Value Part(int index) => b.Slice(table, 0, index, 1);

            var h = b.Add(b.Mul(b.RmsNorm(x, b.Weight(n + "self_attn_norm.weight"), p.Eps), b.Scale(Part(1), 1f, 1f)), Part(0));
            var attended = SelfAttention(b, h, n + "self_attn.", positions, p.Heads, p.KvHeads, p.HeadDim, p.Eps, rope,
                Attention(p.HeadDim, causal: false, window: p.Sliding[il] ? p.AttentionWindow : 0));
            x = b.Add(x, b.Mul(attended, Part(2)));

            h = b.RmsNorm(x, b.Weight(n + "cross_attn_norm.weight"), p.Eps);
            x = b.Add(x, CrossAttention(b, h, condition, n + "cross_attn.", p.Heads, p.KvHeads, p.HeadDim, p.Eps));

            h = b.Add(b.Mul(b.RmsNorm(x, b.Weight(n + "mlp_norm.weight"), p.Eps), b.Scale(Part(4), 1f, 1f)), Part(3));
            x = b.Add(x, b.Mul(Mlp(b, h, n + "mlp."), Part(5)));
        }

        var outTable = b.Add(b.Weight("dit.scale_shift_table"), temb);
        x = b.Add(b.Mul(b.RmsNorm(x, b.Weight("dit.norm_out.weight"), p.Eps), b.Scale(b.Slice(outTable, 0, 1, 1), 1f, 1f)), b.Slice(outTable, 0, 0, 1));
        x = b.Emit1("ConvTranspose1d", [x, b.Weight("dit.proj_out.1.weight"), b.Weight("dit.proj_out.1.bias")],
            Attributes.From(("stride", patch), ("crop_left", 0L), ("crop_right", 0L)));
        b.Output(AudioGenerationGraphs.VelocityOutput, x);
        return b.Build();
    }

    /// <summary>The autoencoder's decoder: latents <c>[T, C]</c> to audio <c>[T · samples per frame, channels]</c>.</summary>
    /// <param name="p">The hyper-parameters.</param>
    /// <param name="weights">Finds a weight by name.</param>
    public static ComputeGraph BuildDecoder(AceStepParameters p, Func<string, WeightInfo?> weights)
    {
        var b = new GraphBuilder(AudioGenerationGraphs.DecodeGraph, weights);
        var x = b.Input(AudioGenerationGraphs.LatentInput, InputRole.Other, DataType.F32, new Shape(Dim.Sym(WellKnown.TokenSymbol), p.LatentChannels));
        x = OobleckDecoderGraph.Build(b, x, "vae.decoder.", p.VaeUpsampling);
        b.Output(AudioGenerationGraphs.AudioOutput, x);
        return b.Build();
    }

    private static Attributes Rope(int headDim, float ropeBase) => Attributes.From(("dims", (long)headDim), ("base", ropeBase), ("style", "neox"));

    private static Attributes Attention(int headDim, bool causal, long window)
        => window > 0
            ? Attributes.From(("scale", 1f / MathF.Sqrt(headDim)), ("causal", causal), ("window", window))
            : Attributes.From(("scale", 1f / MathF.Sqrt(headDim)), ("causal", causal));

    private static Value Linear(GraphBuilder b, Value x, string name) => b.Linear(x, name + ".weight", name + ".bias");

    // A Qwen3 layer: pre-norm self-attention with per-head query and key norms, then a SwiGLU feed-forward layer.
    private static Value Layer(GraphBuilder b, Value x, string n, Value positions, int heads, int kvHeads, int headDim, float eps, Attributes rope, Attributes attention,
        Value? spans = null)
    {
        var h = b.RmsNorm(x, b.Weight(n + "input_layernorm.weight"), eps);
        x = b.Add(x, SelfAttention(b, h, n + "self_attn.", positions, heads, kvHeads, headDim, eps, rope, attention, spans));
        h = b.RmsNorm(x, b.Weight(n + "post_attention_layernorm.weight"), eps);
        return b.Add(x, Mlp(b, h, n + "mlp."));
    }

    private static Value SelfAttention(GraphBuilder b, Value h, string n, Value positions, int heads, int kvHeads, int headDim, float eps, Attributes rope, Attributes attention,
        Value? spans = null)
    {
        var q = b.RmsNorm(b.Reshape(b.MatMul(h, b.Weight(n + "q_proj.weight")), 0, heads, headDim), b.Weight(n + "q_norm.weight"), eps);
        var k = b.RmsNorm(b.Reshape(b.MatMul(h, b.Weight(n + "k_proj.weight")), 0, kvHeads, headDim), b.Weight(n + "k_norm.weight"), eps);
        var v = b.Reshape(b.MatMul(h, b.Weight(n + "v_proj.weight")), 0, kvHeads, headDim);
        q = b.Emit1("Rope", [q, positions], rope);
        k = b.Emit1("Rope", [k, positions], rope);
        var attended = b.Emit1("Attention", spans is null ? [q, k, v] : [q, k, v, spans], attention);
        return b.MatMul(b.Reshape(attended, 0, heads * headDim), b.Weight(n + "o_proj.weight"));
    }

    private static Value CrossAttention(GraphBuilder b, Value h, Value condition, string n, int heads, int kvHeads, int headDim, float eps)
    {
        var q = b.RmsNorm(b.Reshape(b.MatMul(h, b.Weight(n + "q_proj.weight")), 0, heads, headDim), b.Weight(n + "q_norm.weight"), eps);
        var k = b.RmsNorm(b.Reshape(b.MatMul(condition, b.Weight(n + "k_proj.weight")), 0, kvHeads, headDim), b.Weight(n + "k_norm.weight"), eps);
        var v = b.Reshape(b.MatMul(condition, b.Weight(n + "v_proj.weight")), 0, kvHeads, headDim);
        var attended = b.Emit1("Attention", [q, k, v], Attention(headDim, causal: false, window: 0));
        return b.MatMul(b.Reshape(attended, 0, heads * headDim), b.Weight(n + "o_proj.weight"));
    }

    private static Value Mlp(GraphBuilder b, Value h, string n)
        => b.MatMul(b.Mul(b.Silu(b.MatMul(h, b.Weight(n + "gate_proj.weight"))), b.MatMul(h, b.Weight(n + "up_proj.weight"))), b.Weight(n + "down_proj.weight"));

    // Sinusoidal embedding of 1000 · t (cos then sin), two linear layers with SiLU, and the projection to the six modulation vectors ([6, H]).
    private static (Value Embedding, Value Projection) TimestepEmbedding(GraphBuilder b, Value t, string n, int hidden)
    {
        const int Width = 256;
        const int Half = Width / 2;
        var frequencies = new float[Half];
        for (int i = 0; i < Half; i++)
        {
            frequencies[i] = MathF.Exp(-MathF.Log(10000f) * i / Half);
        }

        var table = b.Constant(n + "frequencies", DataType.F32, new Shape(1, Half), System.Runtime.InteropServices.MemoryMarshal.AsBytes(frequencies.AsSpan()).ToArray());
        var angles = b.Mul(b.Reshape(b.Scale(t, 1000f), 1, 1), table);
        var sinusoid = b.Concat(1, b.Unary("Cos", angles), b.Unary("Sin", angles));
        var temb = b.Linear(sinusoid, n + "linear_1.weight", n + "linear_1.bias");
        temb = b.Linear(b.Silu(temb), n + "linear_2.weight", n + "linear_2.bias");
        var projection = b.Linear(b.Silu(temb), n + "time_proj.weight", n + "time_proj.bias");
        return (temb, b.Reshape(projection, 6, hidden));
    }
}
