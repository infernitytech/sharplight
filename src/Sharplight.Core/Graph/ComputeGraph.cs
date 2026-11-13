using System.Text;
using Sharplight.Models;

namespace Sharplight.Graph;

/// <summary>
/// A compute graph: nodes in a valid execution order (topological, preserving side-effect order),
/// its inputs, outputs and the runtime scalars (like <c>n_past</c>) its ops refer to.
/// </summary>
public sealed class ComputeGraph
{
    /// <summary>
    /// Graph name (for example <see cref="WellKnown.DecodeGraph"/>); unique within a <see cref="ModelDefinition"/>.
    /// </summary>
    public string Name { get; }
    /// <summary>
    /// The operations in execution order. Nodes with side effects (<see cref="Node.Writes"/>) keep their relative order.
    /// </summary>
    public IReadOnlyList<Node> Nodes { get; }
    /// <summary>The values the caller supplies for each execution, each with <see cref="Value.Input"/> set.</summary>
    public IReadOnlyList<Value> Inputs { get; }
    /// <summary>The named result values (for example <c>logits</c>); the caller reads them after execution.</summary>
    public IReadOnlyDictionary<string, Value> Outputs { get; }
    /// <summary>
    /// Names of the scalars (for example <see cref="WellKnown.PastScalar"/>) the caller must bind in <see cref="Runtime.RunParameters.Scalars"/> for every execution.
    /// </summary>
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

    /// <summary>
    /// Gets the input that has the given role.
    /// </summary>
    /// <param name="role">The role to look for.</param>
    /// <returns>The first input with that role.</returns>
    /// <exception cref="KeyNotFoundException">The graph has no such input.</exception>
    public Value Input(InputRole role) => Inputs.FirstOrDefault(i => i.Input!.Role == role)
        ?? throw new KeyNotFoundException($"Graph '{Name}' has no input with role {role}.");

    /// <summary>
    /// Gets the input that has the given role, or <see langword="null"/> when the graph has none.
    /// </summary>
    /// <param name="role">The role to look for.</param>
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
