using Sharplight.Graph;

namespace Sharplight.Cpu;

/// <summary>
/// CPU implementation of one row of <see cref="StandardOps.Sample"/>. It follows the device kernel (<c>sample.slang</c>) step by step
/// (same random functions, same top-k threshold, same candidate order and cut rules), so both choose the same token up to
/// floating-point rounding of exp/log and of the probability sums.
/// </summary>
internal static class SamplingReference
{
    /// <summary>Threads per workgroup of the device kernel; candidates beyond the capacity are dropped in its thread-major order.</summary>
    public const int WorkgroupSize = 256;

    public static uint Mix(uint x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    /// <summary>Random stream of one (seed, counter) pair.</summary>
    public static uint Stream(uint seed, uint counter) => Mix(seed ^ Mix(counter ^ 0x6a09e667u));

    /// <summary>Standard Gumbel noise of token <paramref name="i"/> in <paramref name="stream"/>.</summary>
    public static float Gumbel(uint stream, uint i)
    {
        uint h = Mix(stream ^ Mix(i + 0x9e3779b9u));
        float u = ((h >> 8) + 0.5f) * (1f / 16777216f);
        return -MathF.Log(-MathF.Log(u));
    }

    /// <summary>Order-preserving unsigned key of a float.</summary>
    public static uint Key(float f)
    {
        uint u = BitConverter.SingleToUInt32Bits(f);
        return (u & 0x80000000u) != 0 ? ~u : u | 0x80000000u;
    }

    private static float FromKey(uint k) => BitConverter.UInt32BitsToSingle((k & 0x80000000u) != 0 ? k & 0x7FFFFFFFu : ~k);

    /// <summary>Chooses the token of one logits row; <paramref name="logprob"/> is set when the row asks for it (0 otherwise).</summary>
    public static int Sample(ReadOnlySpan<float> x, in SamplingRow row, out float logprob)
    {
        int am = ArgMax(x, out float m);
        int token = row.IsGreedy ? am : SampleFiltered(x, row, m, am);
        logprob = row.WantLogprob ? Logprob(x, token, m) : 0f;
        return token;
    }

    private static int ArgMax(ReadOnlySpan<float> x, out float max)
    {
        int am = 0;
        max = x[0];
        for (int i = 1; i < x.Length; i++)
        {
            if (x[i] > max)
            {
                max = x[i];
                am = i;
            }
        }

        return am;
    }

    private static int SampleFiltered(ReadOnlySpan<float> x, in SamplingRow row, float m, int am)
    {
        float temp = row.Temperature;
        float invT = 1f / temp;
        uint stream = Stream(row.Seed, row.Counter);
        bool nucleus = row.TopP > 0 && row.TopP < 1;
        int k = nucleus && (row.TopK == 0 || row.TopK > SamplingHead.MaxNucleusCandidates) ? SamplingHead.MaxNucleusCandidates : row.TopK;
        uint kthKey = k > 0 && k < x.Length ? KthLargestKey(x, k) : 0;
        if (!nucleus)
        {
            float minLogit = row.MinP > 0 ? m + temp * MathF.Log(row.MinP) : float.NegativeInfinity;
            int best = -1;
            float bestScore = float.NegativeInfinity;
            for (int i = 0; i < x.Length; i++)
            {
                if (Key(x[i]) < kthKey || !(x[i] >= minLogit))
                {
                    continue;
                }

                float s = (x[i] - m) * invT + Gumbel(stream, (uint)i);
                if (best < 0 || s > bestScore)
                {
                    bestScore = s;
                    best = i;
                }
            }

            return best < 0 ? am : best;
        }

        return SampleNucleus(x, row, m, invT, stream, kthKey, am);
    }

    private static int SampleNucleus(ReadOnlySpan<float> x, in SamplingRow row, float m, float invT, uint stream, uint kthKey, int am)
    {
        var candidates = new List<int>();
        for (int i = 0; i < x.Length; i++)
        {
            if (Key(x[i]) >= kthKey)
            {
                candidates.Add(i);
            }
        }

        if (candidates.Count > SamplingHead.MaxNucleusCandidates)
        {
            // The kernel keeps the first candidates in thread-major order (owning thread, then index); with a vocabulary divisible
            // by 4 a thread owns groups of 4 consecutive entries.
            int group = x.Length % 4 == 0 ? 4 : 1;
            candidates.Sort((a, b) =>
            {
                int ta = a / group % WorkgroupSize;
                int tb = b / group % WorkgroupSize;
                return ta != tb ? ta.CompareTo(tb) : a.CompareTo(b);
            });
            candidates.RemoveRange(SamplingHead.MaxNucleusCandidates, candidates.Count - SamplingHead.MaxNucleusCandidates);
        }

        var order = new ulong[candidates.Count];
        for (int c = 0; c < order.Length; c++)
        {
            int i = candidates[c];
            order[c] = ((ulong)~Key(x[i]) << 32) | (uint)i;
        }

        Array.Sort(order);
        var cand = order.Select(o => (int)(uint)o).ToArray();
        int n = cand.Length;
        var p = new float[n];
        float z = 0;
        int cut = n;
        for (int j = 0; j < n; j++)
        {
            p[j] = MathF.Exp((FromKey(Key(x[cand[j]])) - m) * invT);
            z += p[j];
            if (j > 0 && p[j] < row.MinP && cut == n)
            {
                cut = j;
            }
        }

        float mass = 0;
        for (int j = 0; j < cut; j++)
        {
            mass += p[j];
            if (mass >= row.TopP * z)
            {
                cut = j + 1;
                break;
            }
        }

        int best = -1;
        float bestScore = float.NegativeInfinity;
        for (int j = 0; j < cut; j++)
        {
            int i = cand[j];
            float s = (x[i] - m) * invT + Gumbel(stream, (uint)i);
            if (best < 0 || s > bestScore || (s == bestScore && i < best))
            {
                bestScore = s;
                best = i;
            }
        }

        return best < 0 ? am : best;
    }

    /// <summary>Key of the k-th largest value (1-based).</summary>
    private static uint KthLargestKey(ReadOnlySpan<float> x, int k)
    {
        var heap = new PriorityQueue<uint, uint>(k + 1);
        for (int i = 0; i < x.Length; i++)
        {
            uint key = Key(x[i]);
            if (heap.Count < k)
            {
                heap.Enqueue(key, key);
            }
            else if (key > heap.Peek())
            {
                heap.EnqueueDequeue(key, key);
            }
        }

        return heap.Peek();
    }

    private static float Logprob(ReadOnlySpan<float> x, int token, float m)
    {
        double sum = 0;
        foreach (var v in x)
        {
            sum += Math.Exp(v - m);
        }

        return (float)(x[token] - m - Math.Log(sum));
    }
}
