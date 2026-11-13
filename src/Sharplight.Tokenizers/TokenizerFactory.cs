using Sharplight.Tokenizers.Bpe;
using Sharplight.Tokenizers.Spm;
using Sharplight.Tokenizers.Unigram;
using Sharplight.Tokenizers.WordPiece;

namespace Sharplight.Tokenizers;

/// <summary>Creates the tokenizer implementation that matches a <see cref="TokenizerData"/> description.</summary>
public static class TokenizerFactory
{
    /// <summary>Creates a tokenizer for the model family named by <see cref="TokenizerData.Model"/>.</summary>
    /// <param name="data">The tokenizer description; its arrays are used as is and must not be modified afterwards.</param>
    /// <returns>A thread-safe tokenizer.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="data"/> is null.</exception>
    /// <exception cref="NotSupportedException">The model family (or the pre-tokenizer preset of a byte-level BPE) is not supported.</exception>
    /// <exception cref="InvalidDataException">The data lacks what the model family needs (for example scores for unigram).</exception>
    public static ITokenizer Create(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        return data.Model switch
        {
            "gpt2" => new ByteLevelBpeTokenizer(data),
            "spm" or "llama" => new SentencePieceTokenizer(data),
            "wordpiece" => new WordPieceTokenizer(data),
            "unigram" => new UnigramTokenizer(data),
            "bpe" => new RankedBpeTokenizer(data),
            _ => throw new NotSupportedException($"Tokenizer model '{data.Model}' is not supported (supported: gpt2, spm, bpe, wordpiece, unigram)."),
        };
    }
}
