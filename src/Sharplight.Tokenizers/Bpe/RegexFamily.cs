namespace Sharplight.Tokenizers.Bpe;

/// <summary>The regular expression family a <see cref="RegexFamilySplitter"/> implements.</summary>
internal enum RegexFamily
{
    /// <summary>The GPT-2 expression.</summary>
    Gpt2,

    /// <summary>The Llama-3 / Qwen expression family (parameterized by digit grouping and marks).</summary>
    Llama,
}
