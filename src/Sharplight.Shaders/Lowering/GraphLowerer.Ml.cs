using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

// Classical machine learning and text features: decision forests, one-vs-one voting, numeric label encoding, TF-IDF n-grams.
// Their attribute tables (tree nodes, key maps, n-gram pools) become constant device tables of 32-bit words.
public sealed partial class GraphLowerer
{
    private const uint TreeLeaf = 7;
    private const uint TreeMember = 6;
    private const uint TreeNever = 8;
    private const uint TreeAlways = 9;

    /// <summary>
    /// The f32 test equivalent to an f64 threshold test of an f32 value: (mode, threshold) such that comparing the value with the f32
    /// threshold gives the same result as the CPU's comparison in f64 (x ≤ t ⟺ x ≤ the largest float ≤ t, and so on). Equality with a
    /// threshold that is no float, or any test against NaN, becomes a constant outcome.
    /// </summary>
    private static (uint Mode, float Threshold) FloatTest(uint mode, double t)
    {
        if (double.IsNaN(t))
        {
            return (mode == 5 ? TreeAlways : TreeNever, 0f);
        }

        float near = (float)t;
        float down = (double)near > t ? MathF.BitDecrement(near) : near;
        float up = (double)near < t ? MathF.BitIncrement(near) : near;
        bool exact = (double)near == t;
        return mode switch
        {
            0 => (0, down),
            1 => (1, up),
            2 => (2, up),
            3 => (3, down),
            4 => exact ? (4, near) : (TreeNever, 0f),
            5 => exact ? (5, near) : (TreeAlways, 0f),
            _ => (mode, 0f),
        };
    }

    /// <summary>
    /// TreeEnsemble (see <c>StandardOps.TreeEnsemble</c>): one thread per row walks every tree and accumulates the leaf weights of its
    /// targets in its rows of the outputs. Node table: 6 words per node (mode | missing-true bit 8 | set-has-NaN bit 9, feature, true
    /// child, false child, f32 threshold, first member), then the member values, the roots, the leaf offsets, targets and weights.
    /// </summary>
    private void LowerTreeEnsemble(Node n)
    {
        var a = n.Attributes;
        var x = AsDenseF32(Ref(n.Inputs[0]), "tree-x");
        var y = Produce(n.Outputs[0]);
        var hits = Produce(n.Outputs[1]);
        long features = x.Shape[1];
        long targets = a.Get<long>("n_targets");
        string aggregate = a.Get("aggregate", "SUM");
        var roots = a.Get<long[]>("roots");
        var modes = a.Get<long[]>("node_mode");
        var feature = a.Get<long[]>("node_feature");
        var value = a.Get<double[]>("node_value");
        var onTrue = a.Get<long[]>("node_true");
        var onFalse = a.Get<long[]>("node_false");
        var missingTrue = a.Get<long[]>("node_missing_true");
        var memberOffsets = a.Get("member_offsets", Array.Empty<long>());
        var memberValues = a.Get("member_values", Array.Empty<double>());
        var leafOffsets = a.Get<long[]>("leaf_offsets");
        var leafTargets = a.Get<long[]>("leaf_targets");
        var leafWeights = a.Get<float[]>("leaf_weights");
        int nodes = modes.Length;

        var members = new List<uint>();
        var table = new List<uint>(nodes * 6);
        for (int i = 0; i < nodes; i++)
        {
            uint mode = (uint)modes[i];
            float threshold = 0f;
            bool setHasNaN = false;
            int firstMember = members.Count;
            if (mode == TreeMember)
            {
                for (long m = memberOffsets[i]; m < memberOffsets[i + 1]; m++)
                {
                    double member = memberValues[m];
                    setHasNaN |= double.IsNaN(member);
                    if (!double.IsNaN(member) && (double)(float)member == member)
                    {
                        members.Add(BitConverter.SingleToUInt32Bits((float)member));
                    }
                }
            }
            else if (mode != TreeLeaf)
            {
                (mode, threshold) = FloatTest(mode, value[i]);
            }

            if (modes[i] != TreeLeaf && (feature[i] < 0 || feature[i] >= features))
            {
                throw new NotSupportedException($"TreeEnsemble node {i} reads feature {feature[i]} of {features}");
            }

            table.Add(mode | (missingTrue[i] != 0 ? 1u << 8 : 0) | (setHasNaN ? 1u << 9 : 0));
            table.Add((uint)feature[i]);
            table.Add((uint)onTrue[i]);
            table.Add((uint)onFalse[i]);
            table.Add(BitConverter.SingleToUInt32Bits(threshold));
            table.Add((uint)firstMember);
        }

        // A sentinel node: the member range of node i ends at node i + 1's first member.
        table.AddRange([TreeLeaf, 0, 0, 0, 0, (uint)members.Count]);

        long memberBase = table.Count;
        table.AddRange(members);
        long rootBase = table.Count;
        table.AddRange(roots.Select(r => (uint)r));
        long leafOffsetBase = table.Count;
        table.AddRange(leafOffsets.Select(o => (uint)o));
        long leafTargetBase = table.Count;
        table.AddRange(leafTargets.Select(t => (uint)t));
        long leafWeightBase = table.Count;
        table.AddRange(leafWeights.Select(BitConverter.SingleToUInt32Bits));
        var tableRef = ConstTable("tree-ensemble " + n.Outputs[0].Name, [.. table]);

        string combine = aggregate switch
        {
            "MIN" => "p.out1[ob + t] != 0.0 ? min(p.out0[ob + t], w) : w",
            "MAX" => "p.out1[ob + t] != 0.0 ? max(p.out0[ob + t], w) : w",
            _ => "p.out0[ob + t] + w",
        };
        string body = $$"""
              uint xb = idx * {{features}}u, ob = idx * {{targets}}u;
              for (uint t = 0; t < {{targets}}u; t++) { p.out0[ob + t] = 0.0; p.out1[ob + t] = 0.0; }
              for (uint tree = 0; tree < {{roots.Length}}u; tree++) {
                uint node = p.in1[{{rootBase}}u + tree];
                for (;;) {
                  uint word = p.in1[node * 6u];
                  uint mode = word & 255u;
                  if (mode == {{TreeLeaf}}u) { break; }
                  float v = p.in0[xb + p.in1[node * 6u + 1u]];
                  float th = asfloat(p.in1[node * 6u + 4u]);
                  bool go;
                  if (isnan(v)) {
                    go = (word & 256u) != 0u || mode == 5u || mode == {{TreeAlways}}u || (mode == {{TreeMember}}u && (word & 512u) != 0u);
                  } else if (mode == {{TreeMember}}u) {
                    go = false;
                    for (uint m = p.in1[node * 6u + 5u]; m < p.in1[node * 6u + 11u]; m++) { if (asfloat(p.in1[{{memberBase}}u + m]) == v) { go = true; break; } }
                  } else {
                    go = mode == 0u ? v <= th : mode == 1u ? v < th : mode == 2u ? v >= th : mode == 3u ? v > th
                       : mode == 4u ? v == th : mode == 5u ? !(v == th) : mode == {{TreeAlways}}u;
                  }
                  node = p.in1[node * 6u + (go ? 2u : 3u)];
                }
                for (uint k = p.in1[{{leafOffsetBase}}u + node]; k < p.in1[{{leafOffsetBase}}u + node + 1u]; k++) {
                  uint t = p.in1[{{leafTargetBase}}u + k];
                  float w = asfloat(p.in1[{{leafWeightBase}}u + k]);
                  p.out0[ob + t] = {{combine}};
                  p.out1[ob + t] = 1.0;
                }
              }
            {{(aggregate == "AVERAGE" ? $"  for (uint t = 0; t < {targets}u; t++) {{ p.out0[ob + t] = p.out0[ob + t] / {F(roots.Length)}; }}" : "")}}
            """;
        EmitGen("tree_ensemble", [n], [x.Shape[0]], x.Symbol, [x, tableRef], [y, hits], body, outsAreReads: true);
    }

    /// <summary>
    /// OneVsOne (see <c>StandardOps.OneVsOne</c>): one thread per row counts the pairwise votes and, with Platt parameters, couples the
    /// pairwise probabilities as libsvm does (in f32; the CPU provider uses f64).
    /// </summary>
    private void LowerOneVsOne(Node n)
    {
        var a = n.Attributes;
        var d = AsDenseF32(Ref(n.Inputs[0]), "ovo");
        var winner = Produce(n.Outputs[0]);
        int k = (int)a.Get<long>("classes");
        int pairs = k * (k - 1) / 2;
        var probA = a.Get("prob_a", Array.Empty<float>());
        var probB = a.Get("prob_b", Array.Empty<float>());
        bool probabilities = n.Outputs.Length > 1;
        var outs = new List<TensorRef> { winner };
        if (probabilities)
        {
            outs.Add(Produce(n.Outputs[1]));
        }

        string pairwise = probabilities ? $$"""
                  float f = dec * pa[pair] + pb[pair];
                  float pi = f >= 0.0 ? exp(-f) / (1.0 + exp(-f)) : 1.0 / (1.0 + exp(f));
                  pi = clamp(pi, 1e-7, 1.0 - 1e-7);
                  r[i * {{k}} + j] = pi;
                  r[j * {{k}} + i] = 1.0 - pi;
            """ : "";
        string couple = !probabilities ? "" : k == 2 ? $$"""
              p.out1[idx * 2u] = r[1];
              p.out1[idx * 2u + 1u] = r[2];
            """ : $$"""
              float q[{{k * k}}], qp[{{k}}], pr[{{k}}];
              for (int t = 0; t < {{k}}; t++) {
                pr[t] = 1.0 / {{k}}.0;
                q[t * {{k}} + t] = 0.0;
                for (int j = 0; j < t; j++) { q[t * {{k}} + t] += r[j * {{k}} + t] * r[j * {{k}} + t]; q[t * {{k}} + j] = q[j * {{k}} + t]; }
                for (int j = t + 1; j < {{k}}; j++) { q[t * {{k}} + t] += r[j * {{k}} + t] * r[j * {{k}} + t]; q[t * {{k}} + j] = -r[j * {{k}} + t] * r[t * {{k}} + j]; }
              }
              for (int it = 0; it < {{Math.Max(100, k)}}; it++) {
                float pqp = 0.0;
                for (int t = 0; t < {{k}}; t++) {
                  qp[t] = 0.0;
                  for (int j = 0; j < {{k}}; j++) { qp[t] += q[t * {{k}} + j] * pr[j]; }
                  pqp += pr[t] * qp[t];
                }
                float maxError = 0.0;
                for (int t = 0; t < {{k}}; t++) { maxError = max(maxError, abs(qp[t] - pqp)); }
                if (maxError < {{F(0.005f / k)}}) { break; }
                for (int t = 0; t < {{k}}; t++) {
                  float diff = (-qp[t] + pqp) / q[t * {{k}} + t];
                  pr[t] += diff;
                  pqp = (pqp + diff * (diff * q[t * {{k}} + t] + 2.0 * qp[t])) / (1.0 + diff) / (1.0 + diff);
                  for (int j = 0; j < {{k}}; j++) { qp[j] = (qp[j] + diff * q[t * {{k}} + j]) / (1.0 + diff); pr[j] /= 1.0 + diff; }
                }
              }
              for (int t = 0; t < {{k}}; t++) { p.out1[idx * {{k}}u + uint(t)] = pr[t]; }
            """;
        string body = $$"""
              {{(probabilities ? $"const float pa[{pairs}] = {{ {string.Join(", ", probA.Select(F))} }}; const float pb[{pairs}] = {{ {string.Join(", ", probB.Select(F))} }};" : "")}}
              int votes[{{k}}];
              for (int c = 0; c < {{k}}; c++) { votes[c] = 0; }
              {{(probabilities ? $"float r[{k * k}];" : "")}}
              int pair = 0;
              for (int i = 0; i < {{k}}; i++) {
                for (int j = i + 1; j < {{k}}; j++) {
                  float dec = p.in0[idx * {{pairs}}u + uint(pair)];
                  votes[dec > 0.0 ? i : j]++;
            {{pairwise}}
                  pair++;
                }
              }
              int best = 0;
              for (int c = 1; c < {{k}}; c++) { if (votes[c] > votes[best]) { best = c; } }
              p.out0[idx] = best;
            {{couple}}
            """;
        EmitGen("one_vs_one", [n], [d.Shape[0]], d.Symbol, [d], outs, body);
    }

    /// <summary>
    /// The lookup table of a numeric LabelEncode node: keys as the input's type (f32 keys rounded like the CPU provider's, the first of
    /// equal keys wins, NaN keys apart) and values as the output's; null when strings are involved.
    /// </summary>
    private static (uint[] Keys, uint[] Values, uint Default, uint? NaNValue)? LabelEncodeTable(Node node)
    {
        var a = node.Attributes;
        var input = node.Inputs[0].Type;
        var output = node.Output.Type;
        if (!(input == DataType.F32 || input == DataType.I32) || !(output == DataType.F32 || output == DataType.I32)
            || a["keys"] is not double[] keys || a["values"] is not double[] values)
        {
            return null;
        }

        uint Encode(double v) => output == DataType.F32 ? BitConverter.SingleToUInt32Bits((float)v) : (uint)(int)v;
        double fallback = a.TryGetValue("default", out var d) && d is double number ? number : 0.0;
        var seen = new HashSet<double>();
        var outKeys = new List<uint>();
        var outValues = new List<uint>();
        uint? nanValue = null;
        for (int i = 0; i < keys.Length; i++)
        {
            if (double.IsNaN(keys[i]))
            {
                nanValue = Encode(values[i]);
                continue;
            }

            double key = input == DataType.F32 ? (float)keys[i] : keys[i];
            if (input == DataType.I32 && (key != Math.Floor(key) || key < int.MinValue || key > int.MaxValue))
            {
                continue;   // no i32 input equals it
            }

            if (seen.Add(key))
            {
                outKeys.Add(input == DataType.F32 ? BitConverter.SingleToUInt32Bits((float)key) : (uint)(int)key);
                outValues.Add(Encode(values[i]));
            }
        }

        return ([.. outKeys], [.. outValues], Encode(fallback), nanValue);
    }

    /// <summary>Numeric LabelEncode: one thread per element searches the key table (keys first, then the values).</summary>
    private void LowerLabelEncode(Node n)
    {
        var (keys, values, fallback, nanValue) = LabelEncodeTable(n)!.Value;
        var x = EnsureDense(Ref(n.Inputs[0]), "label-encode");
        var y = Produce(n.Output);
        var table = ConstTable("label-encode " + n.Output.Name, [.. keys, .. values]);
        bool isFloat = x.Type == DataType.F32;
        string outType = CType(y.Type);
        string Decode(string bits) => y.Type == DataType.F32 ? $"asfloat({bits})" : $"int({bits})";
        string body = $$"""
              uint bits = {{(isFloat ? "asuint(p.in0[idx])" : "uint(p.in0[idx])")}};
              {{outType}} r = {{Decode($"{fallback}u")}};
              {{(isFloat ? $"if (isnan(p.in0[idx])) {{ r = {Decode($"{nanValue ?? fallback}u")}; }} else" : "")}}
              {
                for (uint i = 0; i < {{keys.Length}}u; i++) {
                  uint key = p.in1[i];
                  if ({{(isFloat ? "asfloat(key) == p.in0[idx]" : "key == bits")}}) { r = {{Decode($"p.in1[{keys.Length}u + i]")}}; break; }
                }
              }
              p.out0[idx] = r;
            """;
        EmitGen("label_encode", [n], x.Rank == 0 ? [1] : x.Shape, x.Symbol, [x, table], [y], body);
    }

    /// <summary>
    /// TfIdf (see <c>StandardOps.TfIdf</c>) in two launches. One thread per (row, pooled n-gram) counts the n-gram's occurrences for
    /// every skip distance (unigrams once); one thread per (row, column) then sums the counts of the n-grams of its column and applies
    /// the weighting. Table: per n-gram its token offset and length, the tokens, then per column the range of its n-grams.
    /// </summary>
    private void LowerTfIdf(Node n)
    {
        var a = n.Attributes;
        var x = EnsureDense(Ref(n.Inputs[0]), "tfidf");
        var y = Produce(n.Output);
        int minGram = (int)a.Get<long>("min_gram");
        int maxGram = (int)a.Get<long>("max_gram");
        int maxSkip = (int)a.Get<long>("max_skip");
        var pool = a.Get<long[]>("pool");
        var ngramCounts = a.Get<long[]>("ngram_counts");
        var ngramIndexes = a.Get<long[]>("ngram_indexes");
        var weights = a.Get("weights", Array.Empty<float>());
        string mode = a.Get<string>("mode");
        long features = a.Get<long>("out_features");

        // The pooled n-grams in [min_gram, max_gram] with their columns; a repeated n-gram keeps its last column, as in the CPU's trie.
        var columnOf = new Dictionary<string, (long[] Tokens, long Column)>();
        int gram = 0;
        for (int length = 1; length <= ngramCounts.Length; length++)
        {
            long start = ngramCounts[length - 1];
            long end = length < ngramCounts.Length ? ngramCounts[length] : pool.Length;
            for (long at = start; at + length <= end; at += length, gram++)
            {
                if (length >= minGram && length <= maxGram && pool[(int)at..(int)(at + length)].All(t => t >= int.MinValue && t <= int.MaxValue))
                {
                    var tokens = pool[(int)at..(int)(at + length)];
                    columnOf[string.Join(",", tokens)] = (tokens, ngramIndexes[gram]);
                }
            }
        }

        var grams = columnOf.Values.Where(g => g.Column >= 0 && g.Column < features).OrderBy(g => g.Column).ToArray();
        var table = new List<uint>();
        long tokenBase = 2L * grams.Length;
        var tokenWords = new List<uint>();
        foreach (var (tokens, _) in grams)
        {
            table.Add((uint)(tokenBase + tokenWords.Count));
            table.Add((uint)tokens.Length);
            tokenWords.AddRange(tokens.Select(t => (uint)(int)t));
        }

        table.AddRange(tokenWords);
        long columnBase = table.Count;
        for (long c = 0, g = 0; c <= features; c++)
        {
            while (g < grams.Length && grams[g].Column < c)
            {
                g++;
            }

            table.Add((uint)g);
        }

        var tableRef = ConstTable("tfidf " + n.Output.Name, [.. table]);
        long rows = x.Rank == 1 ? 1 : x.Shape[0];
        long length2 = x.Shape[^1];
        string? symbol = x.Rank == 1 ? null : x.Symbol;
        long gramCount = Math.Max(1, grams.Length);
        var counts = Temp(DataType.U32, [rows, gramCount], symbol);
        EmitGen("tfidf_count", [n], [rows, gramCount], symbol, [x, tableRef], [counts], $$"""
              uint row = d0, g = d1;
              uint cnt = 0u;
              if (g < {{grams.Length}}u) {
                uint off = p.in1[2u * g], len = p.in1[2u * g + 1u];
                uint xb = row * {{length2}}u;
                for (uint dist = 1u; dist <= {{maxSkip + 1}}u; dist++) {
                  if (len == 1u && dist > 1u) { break; }
                  for (uint s = 0u; s + (len - 1u) * dist < {{length2}}u; s++) {
                    bool match = true;
                    for (uint i = 0u; i < len; i++) { if (uint(p.in0[xb + s + i * dist]) != p.in1[off + i]) { match = false; break; } }
                    if (match) { cnt++; }
                  }
                }
              }
              p.out0[idx] = cnt;
            """);
        string weight = weights.Length > 0 ? "asfloat(p.in2[c])" : "1.0";
        var weightTable = ConstTable("tfidf-weights " + n.Output.Name, weights.Length > 0 ? [.. weights.Select(BitConverter.SingleToUInt32Bits)] : [0u]);
        string result = mode switch
        {
            "TF" => "float(total)",
            "IDF" => $"total > 0u ? {weight} : 0.0",
            _ => $"float(total) * {weight}",
        };
        EmitGen("tfidf", [n], [rows, features], symbol, [counts, tableRef, weightTable], [y], $$"""
              uint row = d0, c = d1;
              uint total = 0u;
              for (uint g = p.in1[{{columnBase}}u + c]; g < p.in1[{{columnBase}}u + c + 1u]; g++) { total += p.in0[row * {{gramCount}}u + g]; }
              p.out0[idx] = {{result}};
            """);
    }
}
