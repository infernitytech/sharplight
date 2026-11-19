namespace Sharplight.Architectures;

/// <summary>
/// Dia's logit processing and token choice for the rows of all codebooks of one step, as the reference's chain: classifier-free guidance, the temperature, the end token only in
/// the first codebook (and forced there when it is the best choice), top-k, top-p, and the end tokens forced in the later codebooks as their delays run out. A codebook that
/// emitted its end token continues with padding.
/// </summary>
internal sealed class DiaTokenSampler
{
    private readonly int _channels;
    private readonly int _vocabulary;
    private readonly int _eos;
    private readonly int _pad;
    private readonly int[] _delayLeft;
    private readonly bool[] _finished;
    private readonly Random _random;
    private readonly DiaGenerationSettings _settings;
    private readonly int _maxGenerationLength;
    private bool _active;

    /// <summary>Starts a generation.</summary>
    /// <param name="channels">Codebooks.</param>
    /// <param name="vocabulary">Entries per codebook head.</param>
    /// <param name="delays">The delay of every codebook.</param>
    /// <param name="eos">The end token.</param>
    /// <param name="pad">The padding token.</param>
    /// <param name="settings">The settings.</param>
    public DiaTokenSampler(int channels, int vocabulary, IReadOnlyList<int> delays, int eos, int pad, DiaGenerationSettings settings)
    {
        _channels = channels;
        _vocabulary = vocabulary;
        _eos = eos;
        _pad = pad;
        _settings = settings;
        _delayLeft = [.. delays];
        _finished = new bool[channels];
        _random = settings.Seed is { } seed ? new Random(seed) : new Random();
        _maxGenerationLength = settings.MaxLength - delays.Max() - 1;
    }

    /// <summary>The first codebook emitted its end token (the audio ends; the later codebooks follow after their delays).</summary>
    public bool Ending => _active;

    /// <summary>Every codebook has emitted its end token.</summary>
    public bool Done => _finished.All(f => f);

    /// <summary>Chooses the next row's tokens.</summary>
    /// <param name="conditional">The conditional logits [channels · vocabulary].</param>
    /// <param name="unconditional">The unconditional logits, or <see langword="null"/> without guidance.</param>
    /// <param name="length">The rows of the sequence so far.</param>
    public int[] Next(ReadOnlySpan<float> conditional, ReadOnlySpan<float> unconditional, int length)
    {
        var scores = new float[_channels * _vocabulary];
        bool guide = !unconditional.IsEmpty && _settings.Guidance > 1f;
        for (int c = 0; c < _channels; c++)
        {
            var row = scores.AsSpan(c * _vocabulary, _vocabulary);
            conditional.Slice(c * _vocabulary, _vocabulary).CopyTo(row);
            if (guide)
            {
                Guide(row, unconditional.Slice(c * _vocabulary, _vocabulary));
            }

            if (_settings.Temperature > 1f)
            {
                for (int v = 0; v < row.Length; v++)
                {
                    row[v] /= _settings.Temperature;
                }
            }

            FilterEnd(row, c);
        }

        if (_settings.Temperature > 0f)
        {
            for (int c = 0; c < _channels; c++)
            {
                var row = scores.AsSpan(c * _vocabulary, _vocabulary);
                if (_settings.TopK > 0 && _settings.TopK < _vocabulary)
                {
                    KeepTopK(row, _settings.TopK);
                }

                if (_settings.TopP < 1f)
                {
                    KeepTopP(row, _settings.TopP);
                }
            }
        }

        // The delay pattern: once the first codebook ends, every codebook ends after its own delay.
        bool first = Argmax(scores.AsSpan(0, _vocabulary)) == _eos;
        _active |= first || length == _maxGenerationLength;
        for (int c = 0; c < _channels; c++)
        {
            if (_active && _delayLeft[c] == 0)
            {
                var row = scores.AsSpan(c * _vocabulary, _vocabulary);
                row.Fill(float.NegativeInfinity);
                row[_eos] = 0f;
            }

            if (_active)
            {
                _delayLeft[c]--;
            }
        }

        var next = new int[_channels];
        for (int c = 0; c < _channels; c++)
        {
            if (_finished[c])
            {
                next[c] = _pad;
                continue;
            }

            var row = scores.AsSpan(c * _vocabulary, _vocabulary);
            next[c] = _settings.Temperature > 0f ? Sample(row) : Argmax(row);
            _finished[c] = next[c] == _eos;
        }

        return next;
    }

    private void Guide(Span<float> row, ReadOnlySpan<float> unconditional)
    {
        float scale = _settings.Guidance;
        if (_settings.GuidanceTopK <= 0 || _settings.GuidanceTopK >= row.Length)
        {
            for (int v = 0; v < row.Length; v++)
            {
                row[v] += (row[v] - unconditional[v]) * scale;
            }

            return;
        }

        var guided = new float[row.Length];
        for (int v = 0; v < row.Length; v++)
        {
            guided[v] = row[v] + ((row[v] - unconditional[v]) * scale);
        }

        var kept = new bool[row.Length];
        foreach (int v in Enumerable.Range(0, row.Length).OrderByDescending(v => guided[v]).ThenBy(v => v).Take(_settings.GuidanceTopK))
        {
            kept[v] = true;
        }

        for (int v = 0; v < row.Length; v++)
        {
            if (!kept[v])
            {
                row[v] = float.NegativeInfinity;
            }
        }
    }

    // Only the first codebook may end the audio; the later codebooks never see the end, padding or start tokens. The first codebook's end token is forced
    // when it is the best choice and suppressed otherwise.
    private void FilterEnd(Span<float> row, int channel)
    {
        row[(channel == 0 ? _eos + 1 : _eos)..].Fill(float.NegativeInfinity);
        if (Argmax(row) == _eos)
        {
            row[.._eos].Fill(float.NegativeInfinity);
        }
        else
        {
            row[_eos] = float.NegativeInfinity;
        }
    }

    private static void KeepTopK(Span<float> row, int k)
    {
        float threshold = KthLargest(row, k);
        for (int v = 0; v < row.Length; v++)
        {
            if (row[v] < threshold)
            {
                row[v] = float.NegativeInfinity;
            }
        }
    }

    private static void KeepTopP(Span<float> row, float p)
    {
        var values = row.ToArray();
        var order = Enumerable.Range(0, values.Length).OrderBy(v => values[v]).ToArray();
        double max = row[order[^1]];
        double sum = 0;
        foreach (int v in order)
        {
            sum += float.IsNegativeInfinity(row[v]) ? 0 : Math.Exp(row[v] - max);
        }

        double cumulative = 0;
        for (int i = 0; i < order.Length - 1; i++)
        {
            int v = order[i];
            cumulative += float.IsNegativeInfinity(row[v]) ? 0 : Math.Exp(row[v] - max) / sum;
            if (cumulative <= 1.0 - p)
            {
                row[v] = float.NegativeInfinity;
            }
        }
    }

    private static float KthLargest(ReadOnlySpan<float> values, int k)
    {
        var copy = values.ToArray();
        Array.Sort(copy);
        return copy[copy.Length - k];
    }

    private static int Argmax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int v = 1; v < row.Length; v++)
        {
            if (row[v] > row[best])
            {
                best = v;
            }
        }

        return best;
    }

    private int Sample(ReadOnlySpan<float> row)
    {
        int best = Argmax(row);
        double max = row[best];
        double total = 0;
        var probabilities = new double[row.Length];
        for (int v = 0; v < row.Length; v++)
        {
            probabilities[v] = float.IsNegativeInfinity(row[v]) ? 0 : Math.Exp(row[v] - max);
            total += probabilities[v];
        }

        double draw = _random.NextDouble() * total;
        for (int v = 0; v < row.Length; v++)
        {
            draw -= probabilities[v];
            if (draw <= 0 && probabilities[v] > 0)
            {
                return v;
            }
        }

        return best;
    }
}
