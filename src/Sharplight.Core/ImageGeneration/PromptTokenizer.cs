using Sharplight.Tokenizers;

namespace Sharplight.ImageGeneration;

/// <summary>Tokenization of diffusion prompts to a fixed context length (shared by the image and video pipelines).</summary>
internal static class PromptTokenizer
{
    /// <summary>
    /// Tokenizes a prompt to exactly the context length: the tokenizer's special tokens (CLIP: start and end; T5: end), truncated to
    /// keep the end token, then padding.
    /// </summary>
    /// <param name="tokenizer">The text encoder's tokenizer.</param>
    /// <param name="contextLength">The encoder's fixed prompt length.</param>
    /// <param name="prompt">The prompt.</param>
    /// <param name="count">Receives the number of prompt tokens before the padding.</param>
    public static int[] Tokenize(ITokenizer tokenizer, int contextLength, string prompt, out int count)
    {
        var ids = tokenizer.Encode(prompt, addSpecial: true, parseSpecial: false).ToList();
        int end = tokenizer.EosId ?? ids[^1];
        if (ids.Count > contextLength)
        {
            ids = [.. ids.Take(contextLength - 1), end];
        }

        count = ids.Count;
        int pad = tokenizer.PadId ?? end;
        while (ids.Count < contextLength)
        {
            ids.Add(pad);
        }

        return [.. ids];
    }
}
