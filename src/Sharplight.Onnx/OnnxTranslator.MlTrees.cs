using Sharplight.Graph;

namespace Sharplight.Onnx;

// ai.onnx.ml decision forests (TreeEnsembleRegressor / TreeEnsembleClassifier of ai.onnx.ml 1-3 and TreeEnsemble of ai.onnx.ml 5) on the IR
// op TreeEnsemble. Base values, binary classification and labels follow onnxruntime: a target no leaf contributed to scores 0 without its
// base value in a regressor and is left out of a classifier's argmax unless base values are given.
internal sealed partial class OnnxTranslator
{
    private static readonly Dictionary<string, long> s_branchModes = new()
    {
        ["BRANCH_LEQ"] = 0, ["BRANCH_LT"] = 1, ["BRANCH_GTE"] = 2, ["BRANCH_GT"] = 3, ["BRANCH_EQ"] = 4, ["BRANCH_NEQ"] = 5, ["BRANCH_MEMBER"] = 6,
        ["LEAF"] = 7,
    };

    private void TreeEnsembleOp(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = X(ins, 0);
        x = x.Rank == 1 ? Rs(x, 1, x.Shape[0].Fixed) : x;
        string name = Nm(n);
        string transform = n.OpType == "TreeEnsemble" ? PostTransformName(n.Int("post_transform", 0)) : n.Str("post_transform", "NONE");
        if (n.OpType == "TreeEnsemble")
        {
            var forest = _b.Emit("TreeEnsemble", [x], ForestV5(n), name + ".forest");
            SetDyn(n, 0, PostTransform(forest[0], transform, name));
            return;
        }

        bool classifier = n.OpType == "TreeEnsembleClassifier";
        string leaf = classifier ? "class" : "target";
        int targets = classifier ? ClassCount(n) : (int)n.Int("n_targets", 1);
        var scoresAndHits = _b.Emit("TreeEnsemble", [x], ForestV3(n, leaf, targets), name + ".forest");
        var (scores, hits) = (scoresAndHits[0], scoresAndHits[1]);
        float[]? baseValues = n.Floats("base_values") ?? (n.Tensor("base_values_as_tensor") is { } b ? OnnxConst.FromTensor(b).F32() : null);
        if (!classifier)
        {
            if (baseValues is { Length: > 0 })
            {
                scores = Emit("Where", [hits, Emit("Add", [scores, F32Const(baseValues, name + ".base")]), F32Const([0f], name + ".zero")]);
            }

            SetDyn(n, 0, PostTransform(scores, transform, name));
            return;
        }

        var classIds = n.Ints("class_ids") ?? [];
        if (classIds.Distinct().Count() == 1 && targets == 2)
        {
            // Binary: one score s for the class the leaves name; two base values give [s + b0, 0], otherwise [-s, s]; class 1 iff s > 0.
            var s = _b.Slice(scores, 1, classIds[0], 1);
            if (baseValues is { Length: > 0 })
            {
                s = Emit("Add", [s, F32Const([baseValues[0]], name + ".base")]);
            }

            var positive = Emit("Greater", [s, F32Const([0f], name + ".zero")]);
            SetDyn(n, 0, ClassLabels(n, Rs(Emit("Cast", [positive], Attributes.From(("type", "i32"))), Fixed(x)[0]), name + ".label"));
            var pairScores = baseValues is { Length: 2 } ? Concat2(s, Emit("Mul", [s, F32Const([0f], name + ".zero")])) : Concat2(_b.Scale(s, -1f), s);
            SetDyn(n, 1, PostTransform(pairScores, transform, name));
            return;
        }

        Value candidates;
        if (baseValues is { Length: > 0 })
        {
            scores = Emit("Add", [scores, F32Const(baseValues, name + ".base")], null, name + ".based");
            candidates = scores;
        }
        else
        {
            candidates = Emit("Where", [hits, scores, F32Const([float.MinValue], name + ".none")]);
        }

        SetDyn(n, 0, ClassLabels(n, ArgMaxRows(candidates, name), name + ".label"));
        SetDyn(n, 1, PostTransform(scores, transform, name));
    }

    private static string PostTransformName(long code) => code switch
    {
        0 => "NONE",
        1 => "SOFTMAX",
        2 => "LOGISTIC",
        3 => "SOFTMAX_ZERO",
        4 => "PROBIT",
        _ => throw new NotSupportedException($"post_transform {code}"),
    };

    /// <summary>IR forest attributes from the (tree id, node id) tables of TreeEnsembleRegressor / TreeEnsembleClassifier.</summary>
    private static Attributes ForestV3(OnnxNode n, string leaf, int targets)
    {
        var trees = n.Ints("nodes_treeids") ?? throw new InvalidDataException($"{n.OpType} needs nodes_treeids");
        var ids = n.Ints("nodes_nodeids")!;
        int count = trees.Length;
        var index = new Dictionary<(long, long), int>();
        for (int i = 0; i < count; i++)
        {
            index[(trees[i], ids[i])] = i;
        }

        var modeNames = n.Attributes["nodes_modes"].Strings;
        var mode = modeNames.Select(m => s_branchModes.TryGetValue(m, out var code) && code != 6 ? code
            : throw new NotSupportedException($"{n.OpType} node mode {m}")).ToArray();
        var values = n.Floats("nodes_values")?.Select(v => (double)v).ToArray() ?? OnnxConst.FromTensor(n.Tensor("nodes_values_as_tensor")!).Floats().Select(v => (double)(float)v).ToArray();
        var trueIds = n.Ints("nodes_truenodeids")!;
        var falseIds = n.Ints("nodes_falsenodeids")!;
        var onTrue = new long[count];
        var onFalse = new long[count];
        var isChild = new bool[count];
        for (int i = 0; i < count; i++)
        {
            if (mode[i] == 7)
            {
                continue;
            }

            onTrue[i] = index[(trees[i], trueIds[i])];
            onFalse[i] = index[(trees[i], falseIds[i])];
            isChild[onTrue[i]] = true;
            isChild[onFalse[i]] = true;
        }

        var roots = new List<long>();
        var seen = new HashSet<long>();
        for (int i = 0; i < count; i++)
        {
            if (!isChild[i] && seen.Add(trees[i]))
            {
                roots.Add(i);
            }
        }

        var leafTrees = n.Ints(leaf + "_treeids") ?? [];
        var leafNodes = n.Ints(leaf + "_nodeids") ?? [];
        var leafTargets = n.Ints(leaf + "_ids") ?? [];
        var leafWeights = n.Floats(leaf + "_weights") ?? OnnxConst.FromTensor(n.Tensor(leaf + "_weights_as_tensor")!).F32();
        var byNode = Enumerable.Range(0, leafTrees.Length).GroupBy(k => index[(leafTrees[k], leafNodes[k])]).ToDictionary(g => g.Key, g => g.ToList());
        var offsets = new long[count + 1];
        var targetList = new List<long>();
        var weightList = new List<float>();
        for (int i = 0; i < count; i++)
        {
            foreach (int k in byNode.GetValueOrDefault(i) ?? [])
            {
                targetList.Add(leafTargets[k]);
                weightList.Add(leafWeights[k]);
            }

            offsets[i + 1] = targetList.Count;
        }

        if (targetList.Any(t => t < 0 || t >= targets))
        {
            throw new InvalidDataException($"{n.OpType}: a leaf names a target outside 0..{targets - 1}");
        }

        return Attributes.From(
            ("n_targets", (long)targets), ("aggregate", n.Str("aggregate_function", "SUM")), ("roots", roots.ToArray()),
            ("node_mode", mode), ("node_feature", n.Ints("nodes_featureids")!), ("node_value", values), ("node_true", onTrue), ("node_false", onFalse),
            ("node_missing_true", n.Ints("nodes_missing_value_tracks_true") ?? new long[count]),
            ("leaf_offsets", offsets), ("leaf_targets", targetList.ToArray()), ("leaf_weights", weightList.ToArray()));
    }

    /// <summary>IR forest attributes from TreeEnsemble (ai.onnx.ml 5): branch nodes first, then one IR node per leaf.</summary>
    private static Attributes ForestV5(OnnxNode n)
    {
        var features = n.Ints("nodes_featureids")!;
        int branches = features.Length;
        var modes = OnnxConst.FromTensor(n.Tensor("nodes_modes")!).Ints();
        var splits = OnnxConst.FromTensor(n.Tensor("nodes_splits")!).Floats();
        var trueIds = n.Ints("nodes_truenodeids")!;
        var falseIds = n.Ints("nodes_falsenodeids")!;
        var trueLeaf = n.Ints("nodes_trueleafs")!;
        var falseLeaf = n.Ints("nodes_falseleafs")!;
        var leafTargets = n.Ints("leaf_targetids")!;
        var leafWeights = OnnxConst.FromTensor(n.Tensor("leaf_weights")!).F32();
        var members = n.Tensor("membership_values") is { } m ? OnnxConst.FromTensor(m).Floats() : [];
        int leaves = leafTargets.Length;
        int count = branches + leaves;
        var mode = new long[count];
        var feature = new long[count];
        var value = new double[count];
        var onTrue = new long[count];
        var onFalse = new long[count];
        var memberOffsets = new long[count + 1];
        var memberValues = new List<double>();
        int memberAt = 0;
        for (int i = 0; i < branches; i++)
        {
            mode[i] = modes[i];
            feature[i] = features[i];
            // Inputs travel as f32, so double thresholds compare after rounding to float (1.2 ≤ 1.2 must hold for x = 1.2).
            value[i] = (float)splits[i];
            onTrue[i] = trueLeaf[i] != 0 ? branches + trueIds[i] : trueIds[i];
            onFalse[i] = falseLeaf[i] != 0 ? branches + falseIds[i] : falseIds[i];
            if (modes[i] == 6)
            {
                // Each set member node takes the next NaN-terminated run of membership_values.
                for (; memberAt < members.Length && !double.IsNaN(members[memberAt]); memberAt++)
                {
                    memberValues.Add((float)members[memberAt]);
                }

                memberAt++;
            }

            memberOffsets[i + 1] = memberValues.Count;
        }

        var leafOffsets = new long[count + 1];
        for (int i = 0; i < count; i++)
        {
            if (i >= branches)
            {
                mode[i] = 7;
                memberOffsets[i + 1] = memberValues.Count;
            }

            leafOffsets[i + 1] = Math.Max(0, i + 1 - branches);
        }

        var roots = n.Ints("tree_roots")!.Select(r => trueLeaf[r] != 0 && falseLeaf[r] != 0 && trueIds[r] == falseIds[r] ? branches + trueIds[r] : r).ToArray();
        return Attributes.From(
            ("n_targets", n.Int("n_targets", 1)), ("aggregate", AggregateName(n.Int("aggregate_function", 1))), ("roots", roots),
            ("node_mode", mode), ("node_feature", feature), ("node_value", value), ("node_true", onTrue), ("node_false", onFalse),
            ("node_missing_true", n.Ints("nodes_missing_value_tracks_true") is { } missing ? [.. missing, .. new long[leaves]] : new long[count]),
            ("member_offsets", memberOffsets), ("member_values", memberValues.ToArray()),
            ("leaf_offsets", leafOffsets), ("leaf_targets", leafTargets), ("leaf_weights", leafWeights));
    }

    private static string AggregateName(long code) => code switch
    {
        0 => "AVERAGE",
        1 => "SUM",
        2 => "MIN",
        3 => "MAX",
        _ => throw new NotSupportedException($"aggregate_function {code}"),
    };
}
