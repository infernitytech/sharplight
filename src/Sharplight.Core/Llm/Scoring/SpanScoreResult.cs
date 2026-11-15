namespace Sharplight.Llm.Scoring;

/// <summary>The scores of the options of a span scoring prompt, in option order.</summary>
/// <param name="Scores">One raw score (logit) per option.</param>
/// <param name="OptionQuestions">The question index of each option.</param>
/// <param name="TokenCount">Tokens of the prompt.</param>
public sealed record SpanScoreResult(IReadOnlyList<float> Scores, IReadOnlyList<int> OptionQuestions, int TokenCount)
{
    /// <summary>The probability of each option among the options of its question (a softmax over their scores).</summary>
    public float[] Probabilities()
    {
        var p = new float[Scores.Count];
        foreach (int q in OptionQuestions.Distinct())
        {
            var members = Enumerable.Range(0, Scores.Count).Where(i => OptionQuestions[i] == q).ToArray();
            float max = members.Max(i => Scores[i]);
            double sum = members.Sum(i => Math.Exp(Scores[i] - max));
            foreach (int i in members)
            {
                p[i] = (float)(Math.Exp(Scores[i] - max) / sum);
            }
        }

        return p;
    }

    /// <summary>The index of the best option of question <paramref name="question"/>.</summary>
    public int Best(int question)
    {
        int best = -1;
        for (int i = 0; i < Scores.Count; i++)
        {
            if (OptionQuestions[i] == question && (best < 0 || Scores[i] > Scores[best]))
            {
                best = i;
            }
        }

        return best;
    }
}
