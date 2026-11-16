using Xunit;

namespace Sharplight.Speech.Tests;

/// <summary>The host-side CTC decodings on hand-made logits: repeats merge, blanks separate repeated tokens, and the beam search sums paths.</summary>
public sealed class CtcDecodingTests
{
    private const int Blank = 0;

    /// <summary>The greedy path merges repeats, drops blanks and keeps a repeat that a blank separates.</summary>
    [Fact]
    public void GreedyMergesRepeatsAndDropsBlanks()
    {
        // Rows: a a _ a b b _ (vocabulary: blank, a, b).
        var logits = Rows([1, 1, 0, 1, 2, 2, 0], vocabulary: 3);
        var tokens = CtcDecoding.Greedy(logits, 3, Blank);
        Assert.Equal([1, 1, 2], tokens.Select(t => t.Token));
        Assert.Equal([0, 3, 4], tokens.Select(t => t.Row));
        Assert.Equal([2, 1, 2], tokens.Select(t => t.Rows));
    }

    /// <summary>A wide beam finds the collapsed sequence whose paths together are likelier than the single best path.</summary>
    [Fact]
    public void BeamSearchSumsThePathsOfASequence()
    {
        // Two rows: the best single path is "_ _" (0.36), but "a" (a a, a _, _ a: 0.16 + 0.24 + 0.24 = 0.64) is likelier as a sequence.
        float[] probabilities =
        [
            0.6f, 0.4f, 0.0001f,
            0.6f, 0.4f, 0.0001f,
        ];
        var logits = probabilities.Select(p => MathF.Log(p)).ToArray();
        Assert.Empty(CtcDecoding.Greedy(logits, 3, Blank));
        var tokens = CtcDecoding.BeamSearch(logits, 3, Blank, beams: 4);
        Assert.Equal([1], tokens.Select(t => t.Token));
    }

    /// <summary>A beam search with one beam on confident logits gives the greedy tokens.</summary>
    [Fact]
    public void BeamSearchAgreesWithGreedyOnConfidentLogits()
    {
        var logits = Rows([1, 1, 0, 1, 2, 2, 0], vocabulary: 3);
        Assert.Equal(CtcDecoding.Greedy(logits, 3, Blank).Select(t => t.Token), CtcDecoding.BeamSearch(logits, 3, Blank, beams: 3).Select(t => t.Token));
    }

    // One row per entry with a large logit on its token.
    private static float[] Rows(int[] best, int vocabulary)
    {
        var logits = new float[best.Length * vocabulary];
        for (int row = 0; row < best.Length; row++)
        {
            logits[(row * vocabulary) + best[row]] = 10f;
        }

        return logits;
    }
}
