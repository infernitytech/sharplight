namespace Sharplight.Llm.Serving;

/// <summary>Per-request sampling configuration. <see cref="Temperature"/> 0 (or <see cref="Greedy"/>) selects the arg-max token.</summary>
public sealed record SamplingParams
{
    public float Temperature { get; init; } = 1f;
    /// <summary>Keep only the K most likely tokens (0 = off).</summary>
    public int TopK { get; init; }
    /// <summary>Nucleus sampling threshold (1 = off).</summary>
    public float TopP { get; init; } = 1f;
    /// <summary>Drop tokens less likely than MinP times the most likely one (0 = off).</summary>
    public float MinP { get; init; }
    /// <summary>Multiplicative (llama.cpp style) penalty on tokens seen in the last <see cref="RepetitionWindow"/> prompt/output tokens (1 = off).</summary>
    public float RepetitionPenalty { get; init; } = 1f;
    public int RepetitionWindow { get; init; } = 256;
    /// <summary>OpenAI style: subtract a flat amount from every token already generated.</summary>
    public float PresencePenalty { get; init; }
    /// <summary>OpenAI style: subtract this times the number of occurrences in the generated text.</summary>
    public float FrequencyPenalty { get; init; }
    public IReadOnlyDictionary<int, float>? LogitBias { get; init; }
    public int? Seed { get; init; }
    public bool Greedy { get; init; }

    public bool IsGreedy => Greedy || Temperature <= 0;
    public static SamplingParams GreedyDefault { get; } = new() { Greedy = true };
}

/// <summary>Log-probability of a token and of the most likely alternatives.</summary>
public sealed record TokenLogprob(int TokenId, string Token, float Logprob, IReadOnlyList<TokenAlternative> TopAlternatives);
public readonly record struct TokenAlternative(int TokenId, string Token, float Logprob);

/// <summary>Stateful sampler of one request: applies penalties/bias over the request's history, then samples. Not thread-safe.</summary>
public sealed class RequestSampler
{
    private readonly SamplingParams _p;
    private readonly Random _rng;
    private readonly bool _hasPenalties;
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

        if (_p.IsGreedy)
        {
            return ArgMax(w);
        }

        float invT = 1f / _p.Temperature;
        bool filtered = _p.TopK > 0 || _p.TopP < 1f || _p.MinP > 0;
        if (!filtered)
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
            for (int i = 0; i < n; i++) { r -= Math.Exp((w[i] - max) * invT); if (r <= 0)
                {
                    return i;
                }
            }
            return n - 1;
        }

        // Candidate selection: top-K (or the 1024 best when only top-p / min-p is active), sorted descending.
        int k = _p.TopK > 0 ? Math.Min(_p.TopK, n) : Math.Min(1024, n);
        SelectTop(w, k, out int cnt);
        var ti = _ti; var tv = _tv;
        var probs = new double[cnt];
        double tot = 0;
        for (int i = 0; i < cnt; i++) { probs[i] = Math.Exp((tv[i] - tv[0]) * invT); tot += probs[i]; }
        int cut = cnt;
        if (_p.MinP > 0)
        {
            for (int i = 1; i < cnt; i++)
            {
                if (probs[i] < _p.MinP * probs[0]) { cut = i; break; }
            }
        }

        if (_p.TopP < 1f)
        {
            double acc = 0;
            for (int i = 0; i < cut; i++) { acc += probs[i] / tot; if (acc >= _p.TopP) { cut = i + 1; break; } }
        }
        double total = 0;
        for (int i = 0; i < cut; i++)
        {
            total += probs[i];
        }

        double rr = _rng.NextDouble() * total;
        for (int i = 0; i < cut; i++) { rr -= probs[i]; if (rr <= 0)
            {
                return ti[i];
            }
        }
        return ti[cut - 1];
    }

    /// <summary>Fills _tv/_ti with the k largest values (descending) using a bounded min-heap.</summary>
    private void SelectTop(ReadOnlySpan<float> w, int k, out int count)
    {
        if (_tv.Length < k) { _tv = new float[k]; _ti = new int[k]; }
        var hv = _tv; var hi = _ti;
        int c = 0;
        for (int i = 0; i < w.Length; i++)
        {
            float v = w[i];
            if (c < k)
            {
                int j = c++;
                while (j > 0) { int pa = (j - 1) >> 1; if (hv[pa] <= v) { break; } hv[j] = hv[pa]; hi[j] = hi[pa]; j = pa; }
                hv[j] = v; hi[j] = i;
            }
            else if (v > hv[0])
            {
                int j = 0;
                while (true)
                {
                    int l = 2 * j + 1; if (l >= c)
                    {
                        break;
                    }

                    int m = l + 1 < c && hv[l + 1] < hv[l] ? l + 1 : l;
                    if (hv[m] >= v)
                    {
                        break;
                    }

                    hv[j] = hv[m]; hi[j] = hi[m]; j = m;
                }
                hv[j] = v; hi[j] = i;
            }
        }
        Array.Sort(hv, hi, 0, c);
        Array.Reverse(hv, 0, c); Array.Reverse(hi, 0, c);
        count = c;
    }

    public static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0; float bv = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > bv) { bv = logits[i]; best = i; }
        }

        return best;
    }

    /// <summary>Log-softmax of the raw logits at <paramref name="token"/> plus the <paramref name="topN"/> most likely tokens.</summary>
    public static (float Logprob, (int Id, float Logprob)[] Top) Logprobs(ReadOnlySpan<float> logits, int token, int topN)
    {
        float max = float.NegativeInfinity;
        foreach (var v in logits)
        {
            if (v > max)
            {
                max = v;
            }
        }

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
