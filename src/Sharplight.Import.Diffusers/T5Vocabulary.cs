namespace Sharplight.Import.Diffusers;

/// <summary>The tokenizer of stand-alone T5 encoder files that come without one.</summary>
internal static class T5Vocabulary
{
    /// <summary>Writes the canonical tokenizer keys of the T5 v1.1 vocabulary.</summary>
    /// <param name="metadata">The canonical metadata.</param>
    /// <exception cref="NotSupportedException">Always, until the vocabulary is embedded.</exception>
    public static void WriteTo(IDictionary<string, object> metadata)
        => throw new NotSupportedException("The T5 encoder file comes without a tokenizer: put its tokenizer.json beside it, or use a T5 encoder GGUF (which includes one).");
}
