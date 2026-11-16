using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Chatterbox's T3 conditioning rows (<c>T3CondEnc</c>) as a graph: the voice-encoder speaker embedding through the speaker projection (one row), the
/// voice's prompt speech tokens embedded with their learned positions and resampled by the perceiver (learned queries cross-attending to the prompt,
/// then attending to themselves, with one shared pre-norm attention block) to a fixed number of rows, and the emotion exaggeration through its
/// projection (one row). The rows open T3's prompt.
/// </summary>
internal static class ChatterboxConditionGraph
{
    /// <summary>The graph name.</summary>
    public const string GraphName = "t3_condition";

    /// <summary>The speaker embedding [1, 256].</summary>
    public const string SpeakerInput = "speaker";

    /// <summary>The prompt speech tokens [prompt].</summary>
    public const string PromptInput = "prompt_tokens";

    /// <summary>The emotion exaggeration [1, 1].</summary>
    public const string EmotionInput = "emotion";

    /// <summary>The conditioning rows [2 + queries, hidden].</summary>
    public const string RowsOutput = "rows";

    private const string P = "t3.cond_enc.";

    /// <summary>Builds the graph.</summary>
    /// <param name="weights">Looks up the weights.</param>
    /// <param name="promptTokens">The number of prompt speech tokens.</param>
    /// <param name="queries">The perceiver's query rows.</param>
    /// <param name="heads">The perceiver's attention heads.</param>
    /// <param name="hidden">T3's width.</param>
    public static ComputeGraph Build(Func<string, WeightInfo?> weights, int promptTokens, int queries, int heads, int hidden)
    {
        var b = new GraphBuilder(GraphName, weights);
        var speaker = b.Input(SpeakerInput, InputRole.Other, DataType.F32, new Shape(1, 256));
        var tokens = b.Input(PromptInput, InputRole.TokenIds, DataType.I32, new Shape(promptTokens));
        var emotion = b.Input(EmotionInput, InputRole.Other, DataType.F32, new Shape(1, 1));

        var speakerRow = b.Linear(speaker, P + "spkr_enc.weight", P + "spkr_enc.bias");
        var positionIds = b.Constant("t3.prompt_positions", DataType.I32, new Shape(promptTokens),
            System.Runtime.InteropServices.MemoryMarshal.AsBytes(Enumerable.Range(0, promptTokens).ToArray().AsSpan()).ToArray());
        var prompt = b.Add(b.Gather(b.Weight("token_embedding.weight"), tokens), b.Gather(b.Weight("t3.speech_pos_emb.emb.weight"), positionIds));
        var query = b.Reshape(b.Weight(P + "perceiver.pre_attention_query"), queries, hidden);
        var resampled = Attend(b, query, prompt, heads, hidden);
        resampled = Attend(b, resampled, resampled, heads, hidden);
        var emotionRow = b.Linear(emotion, P + "emotion_adv_fc.weight");
        var rows = b.Concat(0, speakerRow, resampled, emotionRow);
        b.Output(RowsOutput, rows);
        return b.Build();
    }

    // AttentionBlock2: x1 + proj_out(attention(to_q(norm(x1)), to_k(norm(x2)), to_v(norm(x2)))).
    private static Value Attend(GraphBuilder b, Value x1, Value x2, int heads, int hidden)
    {
        const string A = P + "perceiver.attn.";
        int headDim = hidden / heads;
        long queryRows = x1.Shape[0].Fixed;
        long keyRows = x2.Shape[0].Fixed;
        var n1 = Norm(b, x1, A + "norm");
        var n2 = Norm(b, x2, A + "norm");
        var q = b.Reshape(b.Linear(n1, A + "to_q.weight", A + "to_q.bias"), queryRows, heads, headDim);
        var k = b.Reshape(b.Linear(n2, A + "to_k.weight", A + "to_k.bias"), keyRows, heads, headDim);
        var v = b.Reshape(b.Linear(n2, A + "to_v.weight", A + "to_v.bias"), keyRows, heads, headDim);
        var attention = b.Emit1("Attention", [q, k, v], Attributes.From(("scale", 1f / MathF.Sqrt(headDim)), ("causal", false)));
        return b.Add(x1, b.Linear(b.Reshape(attention, queryRows, hidden), A + "proj_out.weight", A + "proj_out.bias"));
    }

    private static Value Norm(GraphBuilder b, Value x, string module)
        => b.Emit1("LayerNorm", [x, b.Weight(module + ".weight"), b.Weight(module + ".bias")], Attributes.From(("eps", 1e-5f)));
}
