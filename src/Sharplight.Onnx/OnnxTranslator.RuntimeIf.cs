using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// If whose condition is only known at run time: both branches become separate ComputeGraphs (built with their own GraphBuilder) and the
// IR If node runs the selected one (see StandardOps.If). Values the branches read from the enclosing graph are passed in as inputs;
// constants are embedded in the branch graphs.
internal sealed partial class OnnxTranslator
{
    private string _symbolPrefix = "";
    private int _ifCount;

    /// <summary>A dynamic symbol of the graph being built, unique across nested branch graphs.</summary>
    private string NewDynamicSymbol(string hint, Dim bound) => _b.DynamicSymbol(_symbolPrefix + hint, bound);

    private void RuntimeIf(OnnxNode n, TranslatedTensor?[] ins)
    {
        var thenBranch = n.Attributes["then_branch"].G!;
        var elseBranch = n.Attributes["else_branch"].G!;
        var free = FreeVariables(thenBranch).Union(FreeVariables(elseBranch)).Where(_env.ContainsKey).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        var captured = free.Where(name => !_env[name].IsConst).ToList();
        if (captured.Any(name => _env[name] is { LeadingUnits: > 0 } or { TransposedNonZero: true }))
        {
            throw new NotSupportedException("If branches reading a tensor with leading unit dims or a NonZero result");
        }

        string prefix = $"if{_ifCount++}.";
        var thenGraph = BuildBranch(thenBranch, free, prefix + "then.");
        var elseGraph = BuildBranch(elseBranch, free, prefix + "else.");
        var names = Enumerable.Range(0, n.Outputs.Count).Select(k => $"out{k}").ToArray();
        var symbols = new string[names.Length];
        for (int k = 0; k < names.Length; k++)
        {
            var t = thenGraph.Outputs[names[k]];
            var e = elseGraph.Outputs[names[k]];
            if (t.Type != e.Type || t.Rank != e.Rank || t.Shape.Dims.Skip(1).Zip(e.Shape.Dims.Skip(1)).Any(p => p.First != p.Second))
            {
                throw new NotSupportedException($"If output {k}: the branches produce {t.Type}{t.Shape} and {e.Type}{e.Shape}");
            }

            symbols[k] = t.Rank == 0 || t.Shape[0] == e.Shape[0] ? "" : NewDynamicSymbol(prefix + "out", Dim.Max(t.Shape[0], e.Shape[0]));
        }

        var attributes = Attributes.From(("then_branch", thenGraph), ("else_branch", elseGraph), ("inputs", captured.ToArray()), ("outputs", names),
            ("output_symbols", symbols));
        var outputs = _b.Emit("If", [Idx(ins[0]!), .. captured.Select(name => _env[name].V!)], attributes, Nm(n));
        for (int k = 0; k < outputs.Count; k++)
        {
            if (n.Outputs[k] != "")
            {
                SetDyn(n, k, outputs[k]);
            }
        }
    }

    /// <summary>Translates a branch into its own graph; <paramref name="free"/> are the enclosing values it reads.</summary>
    private ComputeGraph BuildBranch(OnnxGraph g, List<string> free, string prefix)
    {
        var saved = free.ToDictionary(name => name, name => _env[name]);
        var (builder, constants, groupScope, heavy, symbolPrefix) = (_b, _constValues, _groupScope, _heavyInGroup, _symbolPrefix);
        _b = new GraphBuilder(prefix.TrimEnd('.'), name => _weights.GetValueOrDefault(name));
        _constValues = new();
        _groupScope = null;
        _symbolPrefix = prefix;
        try
        {
            NextGroup();
            foreach (var name in free)
            {
                var t = saved[name];
                _env[name] = t.IsConst
                    ? new TranslatedTensor { C = t.C, Scalar = t.Scalar }
                    : new TranslatedTensor { V = _b.Input(name, InputRole.Other, t.V!.Type, t.V.Shape), Scalar = t.Scalar };
            }

            RunSubgraph(g, "If branch");
            for (int k = 0; k < g.Outputs.Count; k++)
            {
                var t = Get(g.Outputs[k].Name);
                bool integer = !OnnxType.IsFloat(g.Outputs[k].ElemType);
                var v = t.IsConst ? (integer ? Idx(t) : AsValue(t)) : t.V!;
                if (t.LeadingUnits > 0 || t.TransposedNonZero)
                {
                    throw new NotSupportedException("an If branch returns a tensor with leading unit dims or a NonZero result");
                }

                var type = integer ? DataType.I32 : DataType.F32;
                if (v.Type != type)
                {
                    v = _b.Emit1("Cast", [v], Attributes.From(("type", type.Name)));
                }

                _b.Output($"out{k}", OutputCopy(v, $"{prefix}out{k}"));
            }

            _groupScope?.Dispose();
            return _b.Build();
        }
        finally
        {
            (_b, _constValues, _groupScope, _heavyInGroup, _symbolPrefix) = (builder, constants, groupScope, heavy, symbolPrefix);
            foreach (var (name, t) in saved)
            {
                _env[name] = t;
            }
        }
    }

    /// <summary>Outputs must be produced by a node: inputs, constants and weights are copied (Scale keeps f32 fast paths, Contiguous copies any type).</summary>
    private Value OutputCopy(Value value, string name)
        => value.Kind == ValueKind.Intermediate ? value
            : value.Type == DataType.F32 ? _b.Scale(value, 1f, 0f, name) : _b.Emit1("Contiguous", [value], null, name);

    /// <summary>Names a subgraph (and its nested subgraphs) reads without defining them.</summary>
    private static HashSet<string> FreeVariables(OnnxGraph g)
    {
        var defined = new HashSet<string>(g.Inputs.Select(i => i.Name).Concat(g.Initializers.Select(i => i.Name)), StringComparer.Ordinal);
        var used = new HashSet<string>(StringComparer.Ordinal);
        foreach (var node in g.Nodes)
        {
            used.UnionWith(node.Inputs.Where(i => i != "" && !defined.Contains(i)));
            foreach (var attribute in node.Attributes.Values.Where(a => a.G is not null))
            {
                used.UnionWith(FreeVariables(attribute.G!).Where(i => !defined.Contains(i)));
            }

            defined.UnionWith(node.Outputs);
        }

        used.UnionWith(g.Outputs.Select(o => o.Name).Where(o => !defined.Contains(o)));
        return used;
    }
}
