namespace Sharplight.Studio.Services.Documents;

/// <summary>
/// Fits document text into the tokens left in the context window: what remains after the answer's reserve, the conversation
/// and the user's own text is shared among the attached documents, and a document that does not fit is cut at the end.
/// </summary>
internal static class ContextBudget
{
    /// <summary>Tokens kept free for chat-template markup and safety.</summary>
    public const int TemplateMargin = 64;

    /// <summary>Appended to a shortened document.</summary>
    public const string TruncationNote = "\n[... truncated to fit the context window ...]";

    /// <summary>Tokens available for documents.</summary>
    /// <param name="contextLength">The model's context length.</param>
    /// <param name="answerReserve">Tokens kept for the answer.</param>
    /// <param name="conversationTokens">Tokens of the system prompt, history and the new message's own text.</param>
    public static int Available(int contextLength, int answerReserve, int conversationTokens)
    {
        return Math.Max(0, contextLength - answerReserve - conversationTokens - TemplateMargin);
    }

    /// <summary>Tokens reserved for the answer: the max-token setting, at most a quarter of the context (a quarter when unlimited).</summary>
    public static int AnswerReserve(int contextLength, int maxTokens)
    {
        int quarter = contextLength / 4;
        return maxTokens <= 0 ? quarter : Math.Min(maxTokens, quarter);
    }

    /// <summary>
    /// Shortens <paramref name="text"/> until it takes at most <paramref name="budget"/> tokens (measured with
    /// <paramref name="countTokens"/>), cutting at a line or word boundary.
    /// </summary>
    /// <returns>The text, its token count and whether it was shortened.</returns>
    public static (string Text, int Tokens, bool Truncated) Fit(string text, int budget, Func<string, int> countTokens)
    {
        int tokens = countTokens(text);
        if (tokens <= budget)
        {
            return (text, tokens, false);
        }

        if (budget <= countTokens(TruncationNote))
        {
            return ("", 0, true);
        }

        // Estimate the cut from the token density, then tighten until it fits (a few passes at most).
        int length = (int)(text.Length * ((double)budget / tokens) * 0.95);
        for (int attempt = 0; attempt < 8 && length > 0; attempt++)
        {
            var candidate = Cut(text, length) + TruncationNote;
            int candidateTokens = countTokens(candidate);
            if (candidateTokens <= budget)
            {
                return (candidate, candidateTokens, true);
            }

            length = (int)(length * ((double)budget / candidateTokens) * 0.95);
        }

        return ("", 0, true);
    }

    /// <summary>The first <paramref name="length"/> characters, backed up to the last line break or space.</summary>
    private static string Cut(string text, int length)
    {
        var head = text[..Math.Min(length, text.Length)];
        int boundary = head.LastIndexOfAny(['\n', ' ']);
        return boundary > length / 2 ? head[..boundary] : head;
    }
}
