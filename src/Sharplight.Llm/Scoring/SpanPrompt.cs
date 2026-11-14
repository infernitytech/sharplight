namespace Sharplight.Llm.Scoring;

/// <summary>A tokenized span scoring prompt: its tokens, question spans and option spans (see <see cref="SpanScorer.Encode"/>).</summary>
/// <param name="Tokens">The token ids.</param>
/// <param name="Questions">The question spans, in prompt order.</param>
/// <param name="Options">The option spans, in prompt order; each refers to its question by index.</param>
public sealed record SpanPrompt(int[] Tokens, IReadOnlyList<SpanQuestion> Questions, IReadOnlyList<SpanOption> Options)
{
    /// <summary>Checks that the spans lie inside the tokens, have tokens and that every question has an option.</summary>
    /// <exception cref="ArgumentException">A span is invalid.</exception>
    public void Validate()
    {
        if (Questions.Count == 0 || Options.Count == 0)
        {
            throw new ArgumentException("A span scoring prompt needs at least one question and one option.");
        }

        var answered = new bool[Questions.Count];
        foreach (var q in Questions)
        {
            CheckSpan(q.Start, q.Length, "question");
        }

        foreach (var o in Options)
        {
            CheckSpan(o.Start, o.Length, "option");
            if ((uint)o.Question >= (uint)Questions.Count)
            {
                throw new ArgumentException($"An option refers to question {o.Question}; the prompt has {Questions.Count}.");
            }

            answered[o.Question] = true;
        }

        int open = Array.IndexOf(answered, false);
        if (open >= 0)
        {
            throw new ArgumentException($"Question {open} has no option.");
        }
    }

    private void CheckSpan(int start, int length, string what)
    {
        if (length <= 0 || start < 0 || start + length > Tokens.Length)
        {
            throw new ArgumentException($"A {what} span [{start}, {start + length}) is empty or outside the {Tokens.Length} tokens.");
        }
    }
}
