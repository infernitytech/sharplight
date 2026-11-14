using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>Decision forest evaluation (see <c>StandardOps.TreeEnsemble</c>); rows run in parallel.</summary>
internal sealed unsafe class TreeEnsembleOp : CpuOp
{
    private const int Leaf = 7;
    private const int Member = 6;

    private readonly TRef _x;
    private readonly TRef _y;
    private readonly TRef _hits;
    private readonly int _features;
    private readonly int _targets;
    private readonly string _aggregate;
    private readonly long[] _roots;
    private readonly int[] _mode;
    private readonly int[] _feature;
    private readonly double[] _value;
    private readonly int[] _true;
    private readonly int[] _false;
    private readonly bool[] _missingTrue;
    private readonly HashSet<double>?[] _members;
    private readonly long[] _leafOffsets;
    private readonly int[] _leafTargets;
    private readonly float[] _leafWeights;

    public TreeEnsembleOp(TRef x, TRef y, TRef hits, Attributes a)
    {
        _x = x;
        _y = y;
        _hits = hits;
        _features = (int)x.Dims[1];
        _targets = (int)a.Get<long>("n_targets");
        _aggregate = a.Get("aggregate", "SUM");
        _roots = a.Get<long[]>("roots");
        _mode = Array.ConvertAll(a.Get<long[]>("node_mode"), v => (int)v);
        _feature = Array.ConvertAll(a.Get<long[]>("node_feature"), v => (int)v);
        _value = a.Get<double[]>("node_value");
        _true = Array.ConvertAll(a.Get<long[]>("node_true"), v => (int)v);
        _false = Array.ConvertAll(a.Get<long[]>("node_false"), v => (int)v);
        _missingTrue = Array.ConvertAll(a.Get<long[]>("node_missing_true"), v => v != 0);
        _leafOffsets = a.Get<long[]>("leaf_offsets");
        _leafTargets = Array.ConvertAll(a.Get<long[]>("leaf_targets"), v => (int)v);
        _leafWeights = a.Get<float[]>("leaf_weights");
        var memberOffsets = a.Get("member_offsets", Array.Empty<long>());
        var memberValues = a.Get("member_values", Array.Empty<double>());
        _members = new HashSet<double>?[_mode.Length];
        for (int i = 0; i < _mode.Length; i++)
        {
            if (_mode[i] == Member)
            {
                _members[i] = [.. memberValues[(int)memberOffsets[i]..(int)memberOffsets[i + 1]]];
            }

            if (_mode[i] != Leaf && (_feature[i] < 0 || _feature[i] >= _features))
            {
                throw new NotSupportedException($"TreeEnsemble node {i} reads feature {_feature[i]} of {_features}");
            }
        }
    }

    public override string Name => "TreeEnsemble";

    public override void Run(RunParameters p)
    {
        int rows = (int)_x.Dim(0, p);
        nint x = (nint)_x.P;
        nint y = (nint)_y.P;
        nint hits = (nint)_hits.P;
        CpuParallel.For(rows, r => Predict((float*)x + ((long)r * _features), (float*)y + ((long)r * _targets), (float*)hits + ((long)r * _targets)));
    }

    private void Predict(float* row, float* output, float* hits)
    {
        Span<double> score = stackalloc double[_targets];
        Span<bool> hit = stackalloc bool[_targets];
        score.Clear();
        hit.Clear();
        foreach (long root in _roots)
        {
            int leaf = FindLeaf(row, (int)root);
            for (long k = _leafOffsets[leaf]; k < _leafOffsets[leaf + 1]; k++)
            {
                int t = _leafTargets[k];
                double w = _leafWeights[k];
                score[t] = _aggregate switch
                {
                    "MIN" => hit[t] ? Math.Min(score[t], w) : w,
                    "MAX" => hit[t] ? Math.Max(score[t], w) : w,
                    _ => score[t] + w,
                };
                hit[t] = true;
            }
        }

        for (int t = 0; t < _targets; t++)
        {
            output[t] = (float)(_aggregate == "AVERAGE" ? score[t] / _roots.Length : score[t]);
            hits[t] = hit[t] ? 1f : 0f;
        }
    }

    private int FindLeaf(float* row, int node)
    {
        while (_mode[node] != Leaf)
        {
            double v = row[_feature[node]];
            double threshold = _value[node];
            bool goTrue = (_missingTrue[node] && double.IsNaN(v)) || _mode[node] switch
            {
                0 => v <= threshold,
                1 => v < threshold,
                2 => v >= threshold,
                3 => v > threshold,
                4 => v == threshold,
                5 => v != threshold,
                _ => _members[node]!.Contains(v),
            };
            node = goTrue ? _true[node] : _false[node];
        }

        return node;
    }
}
