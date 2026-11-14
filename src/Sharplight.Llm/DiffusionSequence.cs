namespace Sharplight.Llm;

/// <summary>
/// The state of one masked-diffusion generation (<see cref="DiffusionOptions"/>): the prompt followed by a span that starts as mask tokens,
/// the block and step the schedule is at, and the rule that turns one denoising pass's logits into revealed positions. The caller runs the
/// model's <see cref="Models.WellKnown.DenoiseGraph"/> over <see cref="Tokens"/> and hands every row's logits to <see cref="Apply"/> until
/// <see cref="Done"/>; several sequences can share one pass. Shared by <see cref="DiffusionGenerator"/> and the serving engine.
/// </summary>
internal sealed class DiffusionSequence
{
    private readonly int _maskToken;
    private readonly bool _shiftLogits;
    private readonly DiffusionOptions _options;
    private readonly bool _timestep;
    private readonly int _blockLength;
    private readonly int _blocks;
    private readonly int _stepsPerBlock;
    private readonly List<int> _masked;
    private readonly List<(float Confidence, int Position, int Token)> _candidates;
    private int[] _plan = [];
    private int _block;
    private int _step;

    /// <summary>Starts a generation of <paramref name="span"/> tokens after <paramref name="prompt"/>.</summary>
    /// <exception cref="ArgumentException">The prompt is empty, the span is not positive, or the block schedule does not divide the span or the steps.</exception>
    public DiffusionSequence(ReadOnlySpan<int> prompt, int span, DiffusionOptions options, int maskToken, bool shiftLogits)
    {
        if (prompt.IsEmpty || span <= 0)
        {
            throw new ArgumentException("Diffusion needs a non-empty prompt and at least one token to generate.");
        }

        if (!FitsSchedule(span, options))
        {
            int steps = options.Steps > 0 ? options.Steps : span;
            int block = options.BlockLength > 0 ? options.BlockLength : span;
            throw new ArgumentException($"The span ({span}) must be a multiple of the block length ({block}) and the steps ({steps}) of the block count.");
        }

        _maskToken = maskToken;
        _shiftLogits = shiftLogits;
        _options = options;
        _timestep = options.BlockLength <= 0;
        _blockLength = options.BlockLength > 0 ? options.BlockLength : span;
        _blocks = span / _blockLength;
        _stepsPerBlock = (options.Steps > 0 ? options.Steps : span) / _blocks;
        PromptLength = prompt.Length;
        Span = span;
        Tokens = new int[prompt.Length + span];
        prompt.CopyTo(Tokens);
        Array.Fill(Tokens, maskToken, prompt.Length, span);
        _masked = new List<int>(_blockLength);
        _candidates = new List<(float, int, int)>(_blockLength);
        _block = -1;
        _step = _stepsPerBlock;
        Settle();
    }

    /// <summary>Whether <paramref name="span"/> tokens can be generated on the schedule of <paramref name="options"/>.</summary>
    public static bool FitsSchedule(int span, DiffusionOptions options)
    {
        int steps = options.Steps > 0 ? options.Steps : span;
        int block = options.BlockLength > 0 ? options.BlockLength : span;
        return span > 0 && span % block == 0 && steps % (span / block) == 0;
    }

    /// <summary>The prompt followed by the span (mask tokens where nothing is revealed yet); the input of the next denoising pass.</summary>
    public int[] Tokens { get; }

    /// <summary>Tokens of the prompt.</summary>
    public int PromptLength { get; }

    /// <summary>Tokens of the generated span.</summary>
    public int Span { get; }

    /// <summary>True once every position of the span is revealed.</summary>
    public bool Done => _block >= _blocks;

    /// <summary>Leading tokens of the span that no later step changes (the completed blocks).</summary>
    public int Final => Done ? Span : _block * _blockLength;

    /// <summary>The generated span (mask tokens where nothing is revealed yet).</summary>
    public ReadOnlyMemory<int> Generated => Tokens.AsMemory(PromptLength, Span);

    /// <summary>
    /// Advances one step with the logits of a denoising pass over <see cref="Tokens"/> (<c>Tokens.Length</c> rows of <paramref name="vocab"/>
    /// floats): reveals the masked positions of the current block the schedule picks, most confident first.
    /// </summary>
    public void Apply(ReadOnlySpan<float> logits, int vocab)
    {
        if (Done)
        {
            return;
        }

        int start = PromptLength + _block * _blockLength;
        CollectMasked(start);
        _candidates.Clear();
        foreach (int pos in _masked)
        {
            int row = _shiftLogits ? Math.Max(pos - 1, 0) : pos;
            var (token, confidence) = Predict(logits.Slice(row * vocab, vocab), _options.Unmasking);
            _candidates.Add((confidence, pos, token));
        }

        int count = _timestep ? TimestepCount(_step, _stepsPerBlock, _masked.Count, _options.Eps) : _plan[_step];
        // Most confident first; ties go to the earlier position.
        _candidates.Sort(static (a, b) => a.Confidence != b.Confidence ? b.Confidence.CompareTo(a.Confidence) : a.Position.CompareTo(b.Position));
        for (int i = 0; i < Math.Min(count, _candidates.Count); i++)
        {
            Tokens[_candidates[i].Position] = _candidates[i].Token;
        }

        _step++;
        Settle();
    }

    /// <summary>Moves on to the next block while the current one has no masked position or no step left.</summary>
    private void Settle()
    {
        while (_block < _blocks)
        {
            if (_block >= 0 && _step < _stepsPerBlock && CollectMasked(PromptLength + _block * _blockLength) > 0)
            {
                return;
            }

            _block++;
            _step = 0;
            if (_block < _blocks && !_timestep)
            {
                _plan = TransferPlan(CollectMasked(PromptLength + _block * _blockLength), _stepsPerBlock);
            }
        }
    }

    /// <summary>Fills the masked positions of the block starting at <paramref name="start"/> and returns their count.</summary>
    private int CollectMasked(int start)
    {
        _masked.Clear();
        for (int i = start; i < start + _blockLength; i++)
        {
            if (Tokens[i] == _maskToken)
            {
                _masked.Add(i);
            }
        }

        return _masked.Count;
    }

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
}
