namespace Sharplight.Llm;

/// <summary>Sampling configuration. Temperature 0 = greedy.</summary>
public sealed record SamplingOptions
{
    public float Temperature { get; init; } = 0.7f;
    public int TopK { get; init; } = 40;
    public float TopP { get; init; } = 0.95f;
    public float MinP { get; init; } = 0.0f;
    public float RepetitionPenalty { get; init; } = 1.0f;
    public int RepetitionWindow { get; init; } = 64;
    public int? Seed { get; init; }

    public static SamplingOptions Greedy { get; } = new() { Temperature = 0 };
}

/// <summary>CPU-side token sampler over a logits vector.</summary>
public sealed class Sampler
{
    readonly SamplingOptions _o;
    readonly Random _rng;
    readonly List<int> _history = [];
    (int Id, float Logit)[] _cand = [];

    public Sampler(SamplingOptions options)
    {
        _o = options;
        _rng = options.Seed is int s ? new Random(s) : new Random();
    }

    public void Accept(int token)
    {
        _history.Add(token);
        if (_history.Count > Math.Max(1, _o.RepetitionWindow)) _history.RemoveAt(0);
    }

    public int Sample(ReadOnlySpan<float> logits)
    {
        int n = logits.Length;
        if (_o.Temperature <= 0 && _o.RepetitionPenalty == 1f) return ArgMax(logits);

        if (_cand.Length < n) _cand = new (int, float)[n];
        for (int i = 0; i < n; i++) _cand[i] = (i, logits[i]);
        if (_o.RepetitionPenalty != 1f)
            foreach (var t in _history.Distinct())
            {
                var l = _cand[t].Logit;
                _cand[t].Logit = l > 0 ? l / _o.RepetitionPenalty : l * _o.RepetitionPenalty;
            }
        if (_o.Temperature <= 0)
        {
            int best = 0;
            for (int i = 1; i < n; i++) if (_cand[i].Logit > _cand[best].Logit) best = i;
            return best;
        }

        // top-k via partial sort
        int k = _o.TopK > 0 ? Math.Min(_o.TopK, n) : n;
        var span = _cand.AsSpan(0, n);
        if (k < n)
        {
            span.Sort((a, b) => b.Logit.CompareTo(a.Logit));
            span = span[..k];
        }
        else span.Sort((a, b) => b.Logit.CompareTo(a.Logit));

        float max = span[0].Logit;
        var probs = new float[span.Length];
        double sum = 0;
        for (int i = 0; i < span.Length; i++) { probs[i] = MathF.Exp((span[i].Logit - max) / _o.Temperature); sum += probs[i]; }
        for (int i = 0; i < probs.Length; i++) probs[i] = (float)(probs[i] / sum);

        int cut = probs.Length;
        if (_o.MinP > 0)
            for (int i = 0; i < probs.Length; i++) if (probs[i] < _o.MinP * probs[0]) { cut = Math.Max(1, i); break; }
        if (_o.TopP < 1)
        {
            double acc = 0;
            for (int i = 0; i < cut; i++) { acc += probs[i]; if (acc >= _o.TopP) { cut = i + 1; break; } }
        }
        double total = 0;
        for (int i = 0; i < cut; i++) total += probs[i];
        double r = _rng.NextDouble() * total;
        for (int i = 0; i < cut; i++) { r -= probs[i]; if (r <= 0) return span[i].Id; }
        return span[cut - 1].Id;
    }

    public static int ArgMax(ReadOnlySpan<float> logits)
    {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < logits.Length; i++) if (logits[i] > bv) { bv = logits[i]; best = i; }
        return best;
    }
}
