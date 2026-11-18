using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Documents;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.Search;

/// <summary>The prompt that asks a chat model to answer from numbered passages, and the reading of its citations.</summary>
internal static partial class AnswerPrompt
{
    /// <summary>Tokens kept for the answer.</summary>
    public const int AnswerTokens = 700;

    private const string SystemPrompt = "You answer questions about the user's documents. Use only the numbered passages you are given. "
        + "After each statement cite the passages it comes from in square brackets, like [1] or [2][3]. "
        + "If the passages do not contain the answer, say so briefly instead of guessing.";

    /// <summary>The request for <paramref name="question"/> with as many of <paramref name="passages"/> (best first) as fit the context.</summary>
    /// <param name="question">The question.</param>
    /// <param name="passages">Each passage's label (file and place) and text.</param>
    /// <param name="contextLength">The chat model's context length in tokens.</param>
    /// <param name="countTokens">Counts a text's tokens in the chat model.</param>
    /// <returns>The input and how many passages it carries.</returns>
    public static (ChatInput Input, int Used) Build(string question, IReadOnlyList<(string Label, string Text)> passages, int contextLength, Func<string, int> countTokens)
    {
        int budget = ContextBudget.Available(contextLength, Math.Min(AnswerTokens, contextLength / 4), countTokens(SystemPrompt) + countTokens(question) + 40);
        var context = new StringBuilder();
        int used = 0;
        for (int i = 0; i < passages.Count; i++)
        {
            string block = $"[{i + 1}] {passages[i].Label}\n{passages[i].Text}\n\n";
            int tokens = countTokens(block);
            if (tokens > budget)
            {
                break;
            }

            budget -= tokens;
            context.Append(block);
            used++;
        }

        string user = $"Passages:\n\n{context}Question: {question}";
        var input = new ChatInput(
            [ChatInputMessage.FromText(ChatRole.System, SystemPrompt), ChatInputMessage.FromText(ChatRole.User, user)],
            new SamplingSettings { Temperature = 0.2, TopK = 40, TopP = 0.9, MaxTokens = AnswerTokens },
            EnableThinking: false);
        return (input, used);
    }

    /// <summary>The distinct passage numbers (1-based) the answer cites, in order of first mention and within <paramref name="count"/>.</summary>
    /// <param name="answer">The model's answer.</param>
    /// <param name="count">How many passages were given.</param>
    public static IReadOnlyList<int> Citations(string answer, int count)
    {
        return [.. Citation().Matches(answer).Select(match => int.Parse(match.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture)).Where(number => number >= 1 && number <= count).Distinct()];
    }

    [GeneratedRegex(@"\[(\d{1,3})\]")]
    private static partial Regex Citation();
}
