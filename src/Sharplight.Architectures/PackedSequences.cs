using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>
/// Shared pieces of the packed-sequence graphs (<see cref="WellKnown.EmbedGraph"/>, <see cref="WellKnown.ClassifyGraph"/>): several
/// independent sequences are concatenated along the token axis and processed in one pass without a KV cache. Attention is
/// confined to each sequence by the <see cref="InputRole.RowSpans"/> table; per-sequence results are pooled with
/// <see cref="InputRole.SequenceSpans"/>. Packing (instead of padding to a common length) wastes no compute on padding and lets
/// one compiled graph serve any mix of lengths up to the token budget.
/// </summary>
internal static class PackedSequences
{
    /// <summary>Name of the token-id input [T].</summary>
    public const string TokensInput = "tokens";
    /// <summary>Name of the position input [T, 1] (positions restart at 0 in every sequence).</summary>
    public const string PositionsInput = "positions";
    /// <summary>Name of the optional token-type input [T].</summary>
    public const string TokenTypesInput = "token_types";
    /// <summary>Name of the row-span input [T, 2]: first row and row count each token attends to.</summary>
    public const string RowSpansInput = "row_spans";
    /// <summary>Name of the sequence-span input [B, 2]: first row and length of each sequence.</summary>
    public const string SequenceSpansInput = "sequence_spans";

    /// <summary>Declares the packed-sequence inputs: tokens [T], positions [T, 1], row spans [T, 2], sequence spans [B, 2], optional token types [T].</summary>
    public static PackedSequenceInputs Declare(GraphBuilder b, bool tokenTypes)
    {
        var T = Dim.Sym(WellKnown.TokenSymbol);
        var B = Dim.Sym(WellKnown.SequenceSymbol);
        var tokens = b.Input(TokensInput, InputRole.TokenIds, DataType.I32, new Shape(T));
        var positions = b.Input(PositionsInput, InputRole.Positions, DataType.I32, new Shape(T, 1));
        var types = tokenTypes ? b.Input(TokenTypesInput, InputRole.TokenTypes, DataType.I32, new Shape(T)) : null;
        var rows = b.Input(RowSpansInput, InputRole.RowSpans, DataType.I32, new Shape(T, 2));
        var seqs = b.Input(SequenceSpansInput, InputRole.SequenceSpans, DataType.I32, new Shape(B, 2));
        return new PackedSequenceInputs(tokens, positions, rows, seqs, types);
    }

    /// <summary>Self-attention confined to each packed sequence (optionally causal within it).</summary>
    public static Value Attention(GraphBuilder b, Value q, Value k, Value v, Value rowSpans, float scale, bool causal, float softcap = 0, string? name = null, float[]? alibiSlopes = null)
    {
        var attrs = new List<(string, object)> { ("scale", scale), ("causal", causal) };
        if (softcap > 0)
        {
            attrs.Add(("softcap", softcap));
        }

        if (alibiSlopes is not null)
        {
            attrs.Add(("alibi_slopes", alibiSlopes));
        }

        return b.Emit1("Attention", [q, k, v, rowSpans], Attributes.From([.. attrs]), name);
    }

    /// <summary>One row per sequence: "cls" = first row, "last" = last row, "mean" = average.</summary>
    public static Value Pool(GraphBuilder b, Value hidden, Value sequenceSpans, string pooling, string? name = null) => pooling switch
    {
        "cls" => b.Emit1("SequencePool", [hidden, sequenceSpans], Attributes.From(("mode", "first")), name),
        "last" => b.Emit1("SequencePool", [hidden, sequenceSpans], Attributes.From(("mode", "last")), name),
        "mean" => b.Emit1("SequencePool", [hidden, sequenceSpans], Attributes.From(("mode", "mean")), name),
        _ => throw new NotSupportedException($"Unknown embedding pooling '{pooling}' (mean, cls, last, none)."),
    };

    /// <summary>Emits <see cref="WellKnown.EmbeddingsOutput"/>: pooled per sequence ([B, H]) or per token for pooling "none" ([T, H]), L2-normalized if asked.</summary>
    public static void EmbeddingsOutput(GraphBuilder b, Value hidden, Value sequenceSpans, string pooling, bool normalize)
    {
        var e = pooling == "none" ? hidden : Pool(b, hidden, sequenceSpans, pooling, "pooled");
        if (normalize)
        {
            e = b.L2Norm(e, 1e-24f, "normalized");
        }

        b.Output(WellKnown.EmbeddingsOutput, e);
    }

    /// <summary>Canonical pooling name (accepts a few aliases), or null when unknown.</summary>
    public static string? NormalizePooling(string? pooling) => pooling?.Trim().ToLowerInvariant() switch
    {
        "mean" or "average" or "avg" => "mean",
        "cls" or "first" => "cls",
        "last" or "lasttoken" or "last_token" or "eos" => "last",
        "none" or "token" or "tokens" => "none",
        _ => null,
    };
}
