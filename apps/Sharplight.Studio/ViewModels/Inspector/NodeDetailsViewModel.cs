using Sharplight.Graph;
using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>The details of the selected node: attributes, inputs (with their producers and the weights read), outputs and consumers.</summary>
internal sealed class NodeDetailsViewModel
{
    /// <param name="node">The node.</param>
    /// <param name="ops">Op name per node id, for the producer and consumer links.</param>
    /// <param name="consumers">Ids of the nodes that read the node's outputs.</param>
    /// <param name="weights">The weight index by name, for the stored sizes.</param>
    public NodeDetailsViewModel(NodeInfo node, IReadOnlyDictionary<int, NodeInfo> ops, IReadOnlyList<int> consumers,
        IReadOnlyDictionary<string, PackageWeightInfo> weights)
    {
        Id = node.Id;
        Op = node.Op;
        Group = node.Group;
        Category = OpCategories.Of(node.Op).ToString();
        Attributes = [.. AttributeText.Split(node.Attributes).Select(a => new FactRow(a.Key, a.Value))];
        Inputs = [.. node.InputValues.Select(v => Input(v, ops, weights))];
        Outputs = [.. node.Outputs.Select(GraphValueRow.From)];
        Consumers = [.. consumers.Where(ops.ContainsKey).Select(id => new NodeLinkRow(id, ops[id].Op, ops[id].Group))];
        long weightBytes = node.InputValues.Where(v => v.Kind == ValueKind.Weight)
            .Sum(v => weights.TryGetValue(v.Name, out var w) ? w.ByteSize : 0);
        int weightCount = node.InputValues.Count(v => v.Kind == ValueKind.Weight);
        WeightSummary = weightCount == 0 ? "No weights" : $"{weightCount} weight{(weightCount == 1 ? "" : "s")}, {ByteFormat.Format(weightBytes)}";
    }

    /// <summary>Node id.</summary>
    public int Id { get; }

    /// <summary>Op name.</summary>
    public string Op { get; }

    /// <summary>Placement group (layer), empty when none.</summary>
    public string Group { get; }

    /// <summary>Op family.</summary>
    public string Category { get; }

    /// <summary><c>#12 · layers.3</c>.</summary>
    public string Caption => Group.Length == 0 ? $"#{Id} · {Category}" : $"#{Id} · {Group} · {Category}";

    /// <summary>The attributes as key/value pairs.</summary>
    public IReadOnlyList<FactRow> Attributes { get; }

    /// <summary>True when the node has attributes.</summary>
    public bool HasAttributes => Attributes.Count > 0;

    /// <summary>The inputs in order.</summary>
    public IReadOnlyList<NodeInputRow> Inputs { get; }

    /// <summary>The outputs.</summary>
    public IReadOnlyList<GraphValueRow> Outputs { get; }

    /// <summary>The nodes reading the outputs.</summary>
    public IReadOnlyList<NodeLinkRow> Consumers { get; }

    /// <summary>True when another node reads an output.</summary>
    public bool HasConsumers => Consumers.Count > 0;

    /// <summary>Number and stored size of the weights the node reads.</summary>
    public string WeightSummary { get; }

    private static NodeInputRow Input(NodeInputInfo value, IReadOnlyDictionary<int, NodeInfo> ops, IReadOnlyDictionary<string, PackageWeightInfo> weights)
    {
        string typeShape = $"{value.Type}{value.Shape}";
        return value.Kind switch
        {
            ValueKind.Intermediate => new NodeInputRow("node", value.Name, typeShape, value.Producer,
                ops.TryGetValue(value.Producer, out var producer) ? $"#{producer.Id} {producer.Op}" : ""),
            ValueKind.Weight => new NodeInputRow("weight", value.Name, typeShape, -1,
                weights.TryGetValue(value.Name, out var weight) ? ByteFormat.Format(weight.ByteSize) : ""),
            ValueKind.Input => new NodeInputRow("input", value.Name, typeShape, -1, ""),
            ValueKind.State => new NodeInputRow("state", value.Name, typeShape, -1, ""),
            _ => new NodeInputRow("constant", value.Name, typeShape, -1, ""),
        };
    }
}
