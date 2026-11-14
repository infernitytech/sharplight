using System.Collections;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Runtime;

/// <summary>
/// Orders a graph's nodes by count level so that independent data-dependent counts computed on a device share one segment.
/// </summary>
/// <remarks>
/// A device (not the CPU) learns a data-dependent count (<see cref="ComputeGraph.DynamicSymbols"/>) only after the segment
/// that computes it has finished. A node whose shapes or attributes use such a count therefore runs at a later level than
/// the node that publishes it. Every other node runs at the lowest level its inputs allow, moved up to the earliest level
/// that reads its outputs so that it does not cross a segment boundary for nothing. Nodes that access state or run
/// subgraphs keep their relative order. Within a level the graph order is kept, so the order stays topological.
/// <para>
/// Graphs such as per-class detection heads publish dozens of independent counts (one NonZero and one NonMaxSuppression per
/// class). In graph order each of them ends a segment, so a run pays a device round trip per count; ordered by level they
/// need one round trip per dependency level.
/// </para>
/// </remarks>
internal static class CountLevelSchedule
{
    /// <summary>The diagnostic switch that keeps graph order (one segment per device-published count): <c>SHARPLIGHT_COUNT_LEVELS=0</c>.</summary>
    public const string DisableEnvironmentVariable = "SHARPLIGHT_COUNT_LEVELS";

    /// <summary>Whether the schedule is turned off by <see cref="DisableEnvironmentVariable"/>.</summary>
    public static bool DisabledByEnvironment => Environment.GetEnvironmentVariable(DisableEnvironmentVariable) is "0" or "false";

    /// <summary>
    /// The node indices of <paramref name="graph"/> in execution order and the count level of each node (indexed like
    /// <see cref="ComputeGraph.Nodes"/>), or null when the graph has no data-dependent count published off the CPU.
    /// </summary>
    /// <param name="graph">The graph.</param>
    /// <param name="deviceCount">Whether node i publishes its count on a device that learns it only at the end of its segment.</param>
    public static (int[] Order, int[] Level)? Build(ComputeGraph graph, Func<int, bool> deviceCount)
    {
        if (graph.DynamicSymbols.Count == 0)
        {
            return null;
        }

        var nodes = graph.Nodes;
        var dynamic = graph.DynamicSymbols.Select(d => d.Key).ToHashSet(StringComparer.Ordinal);
        // A count is published by the first node that names it (in an attribute or an output shape).
        var used = new HashSet<string>[nodes.Count];
        var publisher = new Dictionary<string, int>(StringComparer.Ordinal);
        bool anyDevice = false;
        for (int i = 0; i < nodes.Count; i++)
        {
            used[i] = UsedSymbols(nodes[i], dynamic);
            foreach (string s in used[i])
            {
                if (publisher.TryAdd(s, i))
                {
                    anyDevice |= deviceCount(i);
                }
            }
        }

        if (!anyDevice)
        {
            return null;
        }

        var producer = new Dictionary<Value, int>();
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (var o in nodes[i].Outputs)
            {
                producer[o] = i;
            }
        }

        // ASAP levels: after every producer, after the publisher of every count used (one level later when it is published
        // on a device), and not before the previous ordered node.
        var level = new int[nodes.Count];
        var ordered = new bool[nodes.Count];
        int orderFloor = 0;
        for (int i = 0; i < nodes.Count; i++)
        {
            var n = nodes[i];
            int l = 0;
            foreach (var v in n.Inputs)
            {
                if (producer.TryGetValue(v, out int p))
                {
                    l = Math.Max(l, level[p]);
                }
            }

            foreach (string s in used[i])
            {
                if (publisher[s] is int p && p != i)
                {
                    l = Math.Max(l, level[p] + (deviceCount(p) ? 1 : 0));
                }
            }

            // Subgraphs may name any count published before them.
            if (n.Op.Category == OpCategory.ControlFlow)
            {
                foreach (int p in publisher.Values)
                {
                    if (p < i)
                    {
                        l = Math.Max(l, level[p] + (deviceCount(p) ? 1 : 0));
                    }
                }
            }

            ordered[i] = n.Op.Category == OpCategory.ControlFlow || n.Writes.Length > 0 || n.Inputs.Any(v => v.Kind == ValueKind.State);
            if (ordered[i])
            {
                l = Math.Max(l, orderFloor);
                orderFloor = l;
            }

            level[i] = l;
        }

        // Move nodes that publish nothing and keep no order up to the earliest level that reads them.
        var consumers = new List<int>[nodes.Count];
        for (int i = 0; i < nodes.Count; i++)
        {
            foreach (var v in nodes[i].Inputs)
            {
                if (producer.TryGetValue(v, out int p))
                {
                    (consumers[p] ??= []).Add(i);
                }
            }
        }

        var graphOutputs = graph.Outputs.Values.ToHashSet();
        var publishes = publisher.Values.ToHashSet();
        for (int i = nodes.Count - 1; i >= 0; i--)
        {
            if (ordered[i] || publishes.Contains(i) || consumers[i] is null || nodes[i].Outputs.Any(graphOutputs.Contains))
            {
                continue;
            }

            int earliest = consumers[i].Min(c => level[c]);
            level[i] = Math.Max(level[i], earliest);
        }

        var order = Enumerable.Range(0, nodes.Count).OrderBy(i => level[i]).ThenBy(i => i).ToArray();
        return (order, level);
    }

    /// <summary>The data-dependent symbols a node's shapes and attributes refer to.</summary>
    private static HashSet<string> UsedSymbols(Node n, HashSet<string> dynamic)
    {
        var used = new HashSet<string>(StringComparer.Ordinal);
        foreach (var v in n.Inputs.Concat(n.Outputs))
        {
            foreach (var d in v.Shape.Dims)
            {
                AddDim(d, dynamic, used);
            }
        }

        foreach (var value in n.Attributes.Values)
        {
            AddAttribute(value, dynamic, used);
        }

        return used;
    }

    private static void AddDim(Dim d, HashSet<string> dynamic, HashSet<string> used)
    {
        if (d.IsSymbolic)
        {
            foreach (string atom in d.Expression.Atoms())
            {
                if (dynamic.Contains(atom))
                {
                    used.Add(atom);
                }
            }
        }
    }

    /// <summary>Symbols named in an attribute value: in strings (which may hold dimension expressions), dimensions, shapes and collections of them.</summary>
    private static void AddAttribute(object value, HashSet<string> dynamic, HashSet<string> used)
    {
        switch (value)
        {
            case string s:
                foreach (string symbol in dynamic)
                {
                    if (ContainsName(s, symbol))
                    {
                        used.Add(symbol);
                    }
                }

                break;
            case Dim d:
                AddDim(d, dynamic, used);
                break;
            case Shape shape:
                foreach (var d in shape.Dims)
                {
                    AddDim(d, dynamic, used);
                }

                break;
            case IEnumerable items when value.GetType().GetElementType()?.IsPrimitive != true:
                foreach (object? item in items)
                {
                    if (item is not null)
                    {
                        AddAttribute(item, dynamic, used);
                    }
                }

                break;
        }
    }

    /// <summary>Whether <paramref name="text"/> contains <paramref name="name"/> as a whole identifier.</summary>
    private static bool ContainsName(string text, string name)
    {
        for (int at = text.IndexOf(name, StringComparison.Ordinal); at >= 0; at = text.IndexOf(name, at + 1, StringComparison.Ordinal))
        {
            int end = at + name.Length;
            if ((at == 0 || !IsNameChar(text[at - 1])) && (end == text.Length || !IsNameChar(text[end])))
            {
                return true;
            }
        }

        return false;
    }

    private static bool IsNameChar(char c) => char.IsLetterOrDigit(c) || c is '_' or '.' or ':' or '/';
}
