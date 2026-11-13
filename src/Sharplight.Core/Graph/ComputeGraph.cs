using System.Text;
using Sharplight.Models;

namespace Sharplight.Graph;

/// <summary>
/// A compute graph: nodes in a valid execution order (topological, preserving side-effect order),
/// its inputs, outputs and the runtime scalars (like <c>n_past</c>) its ops refer to.
/// </summary>
public sealed class ComputeGraph
{
    public string Name { get; }
    public IReadOnlyList<Node> Nodes { get; }
    public IReadOnlyList<Value> Inputs { get; }
    public IReadOnlyDictionary<string, Value> Outputs { get; }
    public IReadOnlyList<string> RuntimeScalars { get; }

    internal ComputeGraph(string name, IReadOnlyList<Node> nodes, IReadOnlyList<Value> inputs,
        IReadOnlyDictionary<string, Value> outputs, IReadOnlyList<string> runtimeScalars)
    {
        Name = name;
        Nodes = nodes;
        Inputs = inputs;
        Outputs = outputs;
        RuntimeScalars = runtimeScalars;
    }

    /// <summary>All weight values referenced by the graph.</summary>
    public IEnumerable<Value> Weights => Nodes.SelectMany(n => n.Inputs).Where(v => v.Kind == ValueKind.Weight).Distinct();

    /// <summary>All state values referenced by the graph.</summary>
    public IEnumerable<Value> States => Nodes.SelectMany(n => n.Inputs).Where(v => v.Kind == ValueKind.State).Distinct();

    public Value Input(InputRole role) => Inputs.FirstOrDefault(i => i.Input!.Role == role)
        ?? throw new KeyNotFoundException($"Graph '{Name}' has no input with role {role}.");

    public Value? TryInput(InputRole role) => Inputs.FirstOrDefault(i => i.Input!.Role == role);

    /// <summary>Human readable dump, useful for debugging adapters.</summary>
    public string Dump()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"graph {Name}");
        foreach (var i in Inputs)
        {
            sb.AppendLine($"  input  {i} {i.Input}");
        }

        foreach (var n in Nodes)
        {
            sb.AppendLine("  " + n);
        }

        foreach (var (k, v) in Outputs)
        {
            sb.AppendLine($"  output {k} = %{v.Id}");
        }

        return sb.ToString();
    }
}
