using Sharplight.Tokenizers.Bpe;

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
            _ => throw new NotSupportedException($"Tokenizer model '{data.Model}' is not supported (supported: gpt2)."),
        };
    }
}
