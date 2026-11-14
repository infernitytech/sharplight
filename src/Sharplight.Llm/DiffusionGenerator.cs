using Sharplight.Llm.Embeddings;
using Sharplight.Tokenizers;

namespace Sharplight.Llm;

/// <summary>
/// Masked-diffusion decoding (Dream, LLaDA): the prompt is followed by <see cref="GenerationOptions.MaxNewTokens"/> mask tokens, and each
/// step runs the model's bidirectional <see cref="Models.WellKnown.DenoiseGraph"/> over the whole sequence and reveals the masked positions it
/// is most confident about with their most likely token (greedy), as scheduled by <see cref="DiffusionOptions"/>. Obtain it from
/// <see cref="LlmModel.CreateDiffusionGenerator"/>. Not thread-safe.
/// </summary>
public sealed class DiffusionGenerator : IDisposable
{
    private readonly PackedBatchRunner _runner;
    private readonly ITokenizer _tokenizer;
    private readonly int _maskToken;
    private readonly bool _shiftLogits;
    private readonly int _vocab;

    internal DiffusionGenerator(PackedBatchRunner runner, ITokenizer tokenizer, int maskToken, bool shiftLogits)
    {
        _runner = runner;
        _tokenizer = tokenizer;
        _maskToken = maskToken;
        _shiftLogits = shiftLogits;
        _vocab = runner.OutputWidth;
    }

    /// <summary>Longest sequence (prompt plus generated tokens) one pass handles.</summary>
    public int ContextLength => _runner.MaxBatchTokens;

    /// <summary>The mask token the generated span starts from.</summary>
    public int MaskToken => _maskToken;

    /// <summary>
    /// Generates <see cref="GenerationOptions.MaxNewTokens"/> tokens after <paramref name="prompt"/> and returns them, cut before the first
    /// end-of-generation token when <see cref="GenerationOptions.StopAtEog"/> is set. <paramref name="onStep"/> sees the generated span
    /// (mask tokens where nothing is revealed yet) after every step.
    /// </summary>
    /// <exception cref="ArgumentException">The prompt is empty, the sequence does not fit <see cref="ContextLength"/>, or the block schedule does not divide the span or the steps.</exception>
    public int[] Generate(ReadOnlySpan<int> prompt, GenerationOptions options, Action<ReadOnlyMemory<int>>? onStep = null)
    {
        int span = options.MaxNewTokens;
        int length = prompt.Length + span;
        if (prompt.IsEmpty || span <= 0)
        {
            throw new ArgumentException("Diffusion needs a non-empty prompt and at least one token to generate.");
        }

        if (length > ContextLength)
        {
            throw new ArgumentException($"Prompt and generated span ({length} tokens) exceed the context ({ContextLength}).");
        }

        var d = options.Diffusion;
        int steps = d.Steps > 0 ? d.Steps : span;
        int blockLength = d.BlockLength > 0 ? d.BlockLength : span;
        if (span % blockLength != 0 || steps % (span / blockLength) != 0)
        {
            throw new ArgumentException($"The span ({span}) must be a multiple of the block length ({blockLength}) and the steps ({steps}) of the block count.");
        }

        bool timestep = d.BlockLength <= 0;
        int blocks = span / blockLength;
        int stepsPerBlock = steps / blocks;
        var tokens = new int[length];
        prompt.CopyTo(tokens);
        Array.Fill(tokens, _maskToken, prompt.Length, span);
        var masked = new List<int>(blockLength);
        var candidates = new List<(float Confidence, int Position, int Token)>(blockLength);
        for (int block = 0; block < blocks; block++)
        {
            int start = prompt.Length + block * blockLength;
            int end = start + blockLength;
            int[] plan = timestep ? [] : TransferPlan(CountMasked(tokens, start, end), stepsPerBlock);
            for (int step = 0; step < stepsPerBlock; step++)
            {
                masked.Clear();
                for (int i = start; i < end; i++)
                {
                    if (tokens[i] == _maskToken)
                    {
                        masked.Add(i);
                    }
                }

                if (masked.Count == 0)
                {
                    break;
                }

                var logits = _runner.Run([new PackedSequence(tokens)])[0];
                candidates.Clear();
                foreach (int pos in masked)
                {
                    int row = _shiftLogits ? Math.Max(pos - 1, 0) : pos;
                    var (token, confidence) = Predict(logits.AsSpan(row * _vocab, _vocab), d.Unmasking);
                    candidates.Add((confidence, pos, token));
                }

                int count = timestep ? TimestepCount(step, stepsPerBlock, masked.Count, d.Eps) : plan[step];
                // Most confident first; ties go to the earlier position.
                candidates.Sort(static (a, b) => a.Confidence != b.Confidence ? b.Confidence.CompareTo(a.Confidence) : a.Position.CompareTo(b.Position));
                for (int i = 0; i < Math.Min(count, candidates.Count); i++)
                {
                    tokens[candidates[i].Position] = candidates[i].Token;
                }

                onStep?.Invoke(tokens.AsMemory(prompt.Length, span));
            }
        }

        var generated = tokens.AsSpan(prompt.Length);
        if (options.StopAtEog)
        {
            for (int i = 0; i < generated.Length; i++)
            {
                if (_tokenizer.IsEndOfGeneration(generated[i]))
                {
                    return generated[..i].ToArray();
                }
            }
        }

        return generated.ToArray();
    }

    /// <summary>Generates for <paramref name="prompt"/> (encoded with special tokens) and returns the decoded text.</summary>
    public string Generate(string prompt, GenerationOptions options, bool parseSpecial = true)
        => _tokenizer.Decode(Generate(_tokenizer.Encode(prompt, addSpecial: true, parseSpecial: parseSpecial), options));

    /// <summary>The most likely token of one row and the confidence the unmasking rule gives it.</summary>
    private static (int Token, float Confidence) Predict(ReadOnlySpan<float> logits, DiffusionUnmasking unmasking)
    {
        int best = 0;
        float max = float.NegativeInfinity, second = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            float l = logits[i];
            if (l > max)
            {
                second = max;
                max = l;
                best = i;
            }
            else if (l > second)
            {
                second = l;
            }
        }

        double sum = 0, weighted = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double z = logits[i] - max;
            double e = Math.Exp(z);
            sum += e;
            weighted += e * z;
        }

        float confidence = unmasking switch
        {
            DiffusionUnmasking.Margin => (float)((1.0 - Math.Exp(second - max)) / sum),
            // Σ p·log p = Σ e·z / sum − log sum (z = logit − max, e = exp z).
            DiffusionUnmasking.Entropy => (float)(weighted / sum - Math.Log(sum)),
            _ => (float)(1.0 / sum),
        };
        return (best, confidence);
    }

    private int CountMasked(int[] tokens, int start, int end)
    {
        int n = 0;
        for (int i = start; i < end; i++)
        {
            n += tokens[i] == _maskToken ? 1 : 0;
        }

        return n;
    }

    /// <summary>LLaDA's schedule: <paramref name="masked"/> tokens spread over the steps, the remainder on the first ones.</summary>
    private static int[] TransferPlan(int masked, int steps)
    {
        var plan = new int[steps];
        for (int i = 0; i < steps; i++)
        {
            plan[i] = masked / steps + (i < masked % steps ? 1 : 0);
        }

        return plan;
    }

    /// <summary>Dream's timestep schedule: the share 1 − t(s+1)/t(s) of the masked positions, everything at the last step.</summary>
    private static int TimestepCount(int step, int steps, int masked, float eps)
    {
        if (step >= steps - 1)
        {
            return masked;
        }

        float t = 1f - (float)step / steps * (1f - eps);
        float s = 1f - (float)(step + 1) / steps * (1f - eps);
        return (int)(masked * (1f - s / t));
    }

    /// <summary>Releases the compiled graph.</summary>
    public void Dispose() => _runner.Dispose();
}
