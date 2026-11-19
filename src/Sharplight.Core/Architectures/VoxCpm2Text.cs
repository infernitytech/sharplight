using System.Text.RegularExpressions;
using Sharplight.Tokenizers;

namespace Sharplight.Architectures;

/// <summary>
/// The text side of VoxCPM2: whitespace folded as the reference's <c>VoxCPM.generate</c> folds it, and tokens as its <c>mask_multichar_chinese_tokens</c> wrapper makes them
/// (the tokenizer's pieces, a piece of two or more CJK unified ideographs, word-boundary mark dropped, replaced by its single characters' tokens).
/// </summary>
internal sealed partial class VoxCpm2Text
{
    private readonly ITokenizer _tokenizer;
    private readonly Lazy<Dictionary<int, int[]>> _split;

    /// <summary>Creates the text side over the model's tokenizer.</summary>
    /// <param name="tokenizer">The tokenizer.</param>
    public VoxCpm2Text(ITokenizer tokenizer)
    {
        _tokenizer = tokenizer;
        _split = new Lazy<Dictionary<int, int[]>>(BuildSplitMap);
    }

    /// <summary>The text with line breaks and runs of whitespace folded to single spaces.</summary>
    /// <param name="text">The text.</param>
    public static string Fold(string text) => Whitespace().Replace(text.Replace('\n', ' '), " ");

    /// <summary>The tokens of <paramref name="text"/> (no special tokens added).</summary>
    /// <param name="text">The text.</param>
    public int[] Encode(string text)
    {
        var split = _split.Value;
        var result = new List<int>();
        foreach (int id in _tokenizer.Encode(text, addSpecial: false, parseSpecial: false))
        {
            if (split.TryGetValue(id, out var characters))
            {
                result.AddRange(characters);
            }
            else
            {
                result.Add(id);
            }
        }

        return [.. result];
    }

    // Every piece of two or more CJK unified ideographs (U+4E00 to U+9FFF, the word-boundary mark dropped).
    private Dictionary<int, int[]> BuildSplitMap()
    {
        var map = new Dictionary<int, int[]>();
        for (int id = 0; id < _tokenizer.VocabSize; id++)
        {
            string? piece = _tokenizer.TokenString(id);
            if (piece is null)
            {
                continue;
            }

            string clean = piece.Replace("\u2581", "", StringComparison.Ordinal);
            if (clean.Length < 2 || !clean.All(c => c is >= '\u4e00' and <= '\u9fff'))
            {
                continue;
            }

            // As the reference's wrapper: every character's token, the unknown token (0) for characters without one.
            map[id] = [.. clean.Select(c => _tokenizer.TokenId(c.ToString()) ?? 0)];
        }

        return map;
    }

    [GeneratedRegex(@"\s+")]
    private static partial Regex Whitespace();
}
