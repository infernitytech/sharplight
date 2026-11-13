using Sharplight.Tokenizers.Bpe;
using Sharplight.Tokenizers.Spm;

namespace Sharplight.Tokenizers;

public static class TokenizerFactory
{
    /// <summary>Creates a tokenizer for the model family named by <see cref="TokenizerData.Model"/>.</summary>
    public static ITokenizer Create(TokenizerData data)
    {
        ArgumentNullException.ThrowIfNull(data);
        return data.Model switch
        {
            "gpt2" => new ByteLevelBpeTokenizer(data),
            "spm" or "llama" => new SentencePieceTokenizer(data),
            _ => throw new NotSupportedException($"Tokenizer model '{data.Model}' is not supported (supported: gpt2, spm)."),
        };
    }
}
