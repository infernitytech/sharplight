using Sharplight.Graph;

namespace Sharplight.Llm.Serving;

/// <summary>Stateful sampler of one request: applies penalties/bias over the request's history, then samples. Not thread-safe.</summary>
internal sealed class RequestSampler
{
    /// <summary>Candidates a top-p or min-p cut considers when no top-k bounds the set.</summary>
    private const int DefaultCandidateLimit = 1024;

    private readonly SamplingParams _p;
    private readonly Random _rng;
    private readonly bool _hasPenalties;
    private readonly uint _deviceSeed;
    private float[] _work = [];
    private float[] _tv = [];
    private int[] _ti = [];
    private readonly Dictionary<int, int> _gen = [];
    private readonly Dictionary<int, int> _winCounts = [];
    private readonly Queue<int> _win = new();

    public RequestSampler(SamplingParams p, ReadOnlySpan<int> prompt)
    {
        _p = p;
        _rng = p.Seed is int s ? new Random(s) : new Random();
        _deviceSeed = unchecked((uint)(p.Seed ?? Random.Shared.Next()));
        _hasPenalties = p.RepetitionPenalty != 1f || p.PresencePenalty != 0 || p.FrequencyPenalty != 0;
        if (p.RepetitionPenalty != 1f)
        {
            foreach (var t in prompt[Math.Max(0, prompt.Length - p.RepetitionWindow)..])
            {
                Push(t);
            }
        }
    }

    private void Push(int t)
    {
        _win.Enqueue(t);
        _winCounts[t] = _winCounts.GetValueOrDefault(t) + 1;
        while (_win.Count > Math.Max(1, _p.RepetitionWindow))
        {
            int o = _win.Dequeue();
            if (--_winCounts[o] == 0)
            {
                _winCounts.Remove(o);
            }
        }
    }

    /// <summary>Records a generated token (after it was sampled).</summary>
    public void Accept(int token)
    {
        if (!_hasPenalties)
        {
            return;
        }

        _gen[token] = _gen.GetValueOrDefault(token) + 1;
        if (_p.RepetitionPenalty != 1f)
        {
            Push(token);
        }
    }

    /// <summary>
    /// The device sampling row for the <paramref name="counter"/>-th sampled token, or null when the request needs the CPU path:
    /// penalties, logit bias, alternatives for logprobs, or a top-p cut over more than <see cref="SamplingHead.MaxNucleusCandidates"/>
    /// candidates (the CPU sampler then uses the full top-k set).
    /// </summary>
    /// <param name="counter">Number of tokens the request sampled before.</param>
    /// <param name="logprob">Whether the chosen token's log-probability is reported.</param>
    /// <param name="topLogprobs">Number of alternatives reported with it.</param>
    public SamplingRow? DeviceRow(int counter, bool logprob, int topLogprobs)
    {
        if (_hasPenalties || _p.LogitBias is not null || (logprob && topLogprobs > 0))
        {
            return null;
        }

        if (_p.IsGreedy)
        {
            return SamplingRow.Greedy(logprob);
        }

        bool nucleus = _p.TopP < 1f;
        if (nucleus && (_p.TopP <= 0 || _p.TopK > SamplingHead.MaxNucleusCandidates))
        {
            return null;
        }

        return new SamplingRow
        {
            Temperature = _p.Temperature,
            TopK = _p.TopK,
            TopP = _p.TopP,
            MinP = _p.MinP,
            Seed = _deviceSeed,
            Counter = unchecked((uint)counter),
            WantLogprob = logprob,
        };
    }

    /// <summary>Picks the next token from <paramref name="logits"/> (not modified) after applying bias and penalties.</summary>
    public int Sample(ReadOnlySpan<float> logits)
    {
        int n = logits.Length;
        if (_p.IsGreedy && !_hasPenalties && _p.LogitBias is null)
        {
            return ArgMax(logits);
        }

        if (_work.Length < n)
        {
            _work = new float[n];
        }

        var w = _work.AsSpan(0, n);
        logits.CopyTo(w);
        ApplyBiasAndPenalties(w);
        if (_p.IsGreedy)
        {
            return ArgMax(w);
        }

        float invT = 1f / _p.Temperature;
        bool filtered = _p.TopK > 0 || _p.TopP < 1f || _p.MinP > 0;
        return filtered ? SampleFiltered(w, invT) : SampleFromFullDistribution(w, invT);
    }

    private void ApplyBiasAndPenalties(Span<float> w)
    {
        int n = w.Length;
        if (_p.LogitBias is { } bias)
        {
            foreach (var (id, b) in bias)
            {
                if ((uint)id < (uint)n)
                {
                    w[id] += b;
                }
            }
        }

        if (_p.RepetitionPenalty != 1f)
        {
            foreach (var t in _winCounts.Keys)
            {
                if ((uint)t < (uint)n)
                {
                    w[t] = w[t] > 0 ? w[t] / _p.RepetitionPenalty : w[t] * _p.RepetitionPenalty;
                }
            }
        }

        if (_p.PresencePenalty != 0 || _p.FrequencyPenalty != 0)
        {
            foreach (var (t, c) in _gen)
            {
                if ((uint)t < (uint)n)
                {
                    w[t] -= _p.PresencePenalty + _p.FrequencyPenalty * c;
                }
            }
        }
    }

    /// <summary>Samples from the softmax over every token (no top-k, top-p or min-p cut).</summary>
    private int SampleFromFullDistribution(ReadOnlySpan<float> w, float invT)
    {
        float max = float.NegativeInfinity;
        foreach (var v in w)
        {
            if (v > max)
            {
                max = v;
            }
        }

        double sum = 0;
        foreach (var v in w)
        {
            sum += Math.Exp((v - max) * invT);
        }

        double r = _rng.NextDouble() * sum;
        for (int i = 0; i < w.Length; i++)
        {
            r -= Math.Exp((w[i] - max) * invT);
            if (r <= 0)
            {
                return i;
            }
        }

        return w.Length - 1;
    }

    /// <summary>Samples among the top-k candidates, cut by min-p and then top-p.</summary>
    private int SampleFiltered(ReadOnlySpan<float> w, float invT)
    {
        int k = _p.TopK > 0 ? Math.Min(_p.TopK, w.Length) : Math.Min(DefaultCandidateLimit, w.Length);
        SelectTop(w, k, out int candidates);
        var ids = _ti;
        var values = _tv;
        var probs = new double[candidates];
        double tot = 0;
        for (int i = 0; i < candidates; i++)
        {
            probs[i] = Math.Exp((values[i] - values[0]) * invT);
            tot += probs[i];
        }

        int cut = candidates;
        if (_p.MinP > 0)
        {
            for (int i = 1; i < candidates; i++)
            {
                if (probs[i] < _p.MinP * probs[0])
                {
                    cut = i;
                    break;
                }
            }
        }

        if (_p.TopP < 1f)
        {
            double acc = 0;
            for (int i = 0; i < cut; i++)
            {
                acc += probs[i] / tot;
                if (acc >= _p.TopP)
                {
                    cut = i + 1;
                    break;
                }
            }
        }

        double total = 0;
        for (int i = 0; i < cut; i++)
        {
            total += probs[i];
        }

        double rr = _rng.NextDouble() * total;
        for (int i = 0; i < cut; i++)
        {
            rr -= probs[i];
            if (rr <= 0)
            {
                return ids[i];
            }
        }

        return ids[cut - 1];
    }

    /// <summary>Fills _tv/_ti with the k largest values (descending) using a bounded min-heap.</summary>
    private void SelectTop(ReadOnlySpan<float> w, int k, out int count)
    {
        if (_tv.Length < k)
        {
            _tv = new float[k];
            _ti = new int[k];
        }

        var heapValues = _tv;
        var heapIds = _ti;
        int size = 0;
        for (int i = 0; i < w.Length; i++)
        {
            float v = w[i];
            if (size < k)
            {
                SiftUp(heapValues, heapIds, size++, v, i);
            }
            else if (v > heapValues[0])
            {
                SiftDownReplaceRoot(heapValues, heapIds, size, v, i);
            }
        }

        Array.Sort(heapValues, heapIds, 0, size);
        Array.Reverse(heapValues, 0, size);
        Array.Reverse(heapIds, 0, size);
        count = size;
    }

    /// <summary>Inserts (<paramref name="value"/>, <paramref name="id"/>) at the new leaf <paramref name="leaf"/> of a min-heap.</summary>
    private static void SiftUp(float[] values, int[] ids, int leaf, float value, int id)
    {
        int j = leaf;
        while (j > 0)
        {
            int parent = (j - 1) >> 1;
            if (values[parent] <= value)
            {
                break;
            }

            values[j] = values[parent];
            ids[j] = ids[parent];
            j = parent;
        }

        values[j] = value;
        ids[j] = id;
    }

    /// <summary>Replaces the root of a min-heap of <paramref name="size"/> entries with (<paramref name="value"/>, <paramref name="id"/>).</summary>
    private static void SiftDownReplaceRoot(float[] values, int[] ids, int size, float value, int id)
    {
        int j = 0;
        while (true)
        {
            int left = 2 * j + 1;
            if (left >= size)
            {
                break;
            }

            int smaller = left + 1 < size && values[left + 1] < values[left] ? left + 1 : left;
            if (values[smaller] >= value)
            {
                break;
            }

            values[j] = values[smaller];
            ids[j] = ids[smaller];
            j = smaller;
        }

        values[j] = value;
        ids[j] = id;
    }

    public static int ArgMax(ReadOnlySpan<float> logits) => LogitsMath.ArgMax(logits);

    /// <summary>Log-softmax of the raw logits at <paramref name="token"/> plus the <paramref name="topN"/> most likely tokens.</summary>
    public static (float Logprob, (int Id, float Logprob)[] Top) Logprobs(ReadOnlySpan<float> logits, int token, int topN)
    {
        float max = LogitsMath.Max(logits);
        double sum = 0;
        foreach (var v in logits)
        {
            sum += Math.Exp(v - max);
        }

        float lse = max + (float)Math.Log(sum);
        var top = new List<(int, float)>();
        if (topN > 0)
        {
            var s = new RequestSampler(SamplingParams.GreedyDefault, []);
            s.SelectTop(logits, Math.Min(topN, logits.Length), out int c);
            for (int i = 0; i < c; i++)
            {
                top.Add((s._ti[i], s._tv[i] - lse));
            }
        }

        return (logits[token] - lse, [.. top]);
    }
}
