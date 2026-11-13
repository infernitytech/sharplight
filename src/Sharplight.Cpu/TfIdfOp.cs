using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>N-gram counting with TF / IDF / TF-IDF weighting over i32 token rows (see <c>StandardOps.TfIdf</c>).</summary>
internal sealed unsafe class TfIdfOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly int _minGram;
    private readonly int _maxGram;
    private readonly int _maxSkip;
    private readonly float[] _weights;
    private readonly string _mode;
    private readonly int _features;
    private readonly Trie _root = new();

    public TfIdfOp(TRef x, TRef y, int minGram, int maxGram, int maxSkip, long[] pool, long[] ngramCounts, long[] ngramIndexes, float[] weights, string mode, int features)
    {
        _x = x;
        _y = y;
        _minGram = minGram;
        _maxGram = maxGram;
        _maxSkip = maxSkip;
        _weights = weights;
        _mode = mode;
        _features = features;

        // N-gram k of the pool (in pool order, over all lengths) maps to column ngramIndexes[k]; only lengths in [min, max] are counted.
        int gram = 0;
        for (int length = 1; length <= ngramCounts.Length; length++)
        {
            long start = ngramCounts[length - 1];
            long end = length < ngramCounts.Length ? ngramCounts[length] : pool.Length;
            for (long at = start; at + length <= end; at += length, gram++)
            {
                if (length < minGram || length > maxGram)
                {
                    continue;
                }

                var node = _root;
                for (int k = 0; k < length; k++)
                {
                    node = node.Child(pool[at + k]);
                }

                node.Column = (int)ngramIndexes[gram];
            }
        }
    }

    public override string Name => "TfIdf";

    public override void Run(RunParameters p)
    {
        long rows = _x.Rank == 1 ? 1 : _x.Dim(0, p);
        int length = (int)_x.Dims[^1];
        float* output = (float*)_y.P;
        var counts = new int[_features];
        for (long r = 0; r < rows; r++)
        {
            Array.Clear(counts);
            Count((int*)_x.P + (r * length), length, counts);
            float* o = output + (r * _features);
            for (int c = 0; c < _features; c++)
            {
                float weight = _weights.Length > 0 ? _weights[c] : 1f;
                o[c] = _mode switch
                {
                    "TF" => counts[c],
                    "IDF" => counts[c] > 0 ? weight : 0f,
                    _ => counts[c] * weight,
                };
            }
        }
    }

    private void Count(int* row, int length, int[] counts)
    {
        int startSize = _minGram;
        for (int distance = 1; distance <= _maxSkip + 1; distance++)
        {
            for (int start = 0; start < length && start + (distance * (startSize - 1)) < length; start++)
            {
                var node = _root;
                int size = 1;
                for (int item = start; size <= _maxGram && item < length; item += distance, size++)
                {
                    if (node.Children is null || !node.Children.TryGetValue(row[item], out node))
                    {
                        break;
                    }

                    if (size >= startSize && node.Column >= 0)
                    {
                        counts[node.Column]++;
                    }
                }
            }

            // Unigrams do not depend on the skip distance: count them once.
            if (startSize == 1)
            {
                startSize = 2;
                if (startSize > _maxGram)
                {
                    break;
                }
            }
        }
    }

    private sealed class Trie
    {
        public Dictionary<long, Trie>? Children;
        public int Column = -1;

        public Trie Child(long token)
        {
            Children ??= [];
            if (!Children.TryGetValue(token, out var child))
            {
                child = new Trie();
                Children[token] = child;
            }

            return child;
        }
    }
}
