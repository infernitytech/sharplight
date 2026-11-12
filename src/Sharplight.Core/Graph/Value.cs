using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Graph;

public enum ValueKind
{
    /// <summary>Produced by a node inside the graph.</summary>
    Intermediate,
    /// <summary>Supplied by the caller for each execution (token ids, positions, embeddings, ...).</summary>
    Input,
    /// <summary>Model parameter, backed by a <see cref="WeightInfo"/>.</summary>
    Weight,
    /// <summary>Persistent per-sequence state (KV cache, recurrent state) that survives across executions.</summary>
    State,
    /// <summary>A small constant embedded in the graph.</summary>
    Constant,
}

/// <summary>An SSA value (tensor) in a <see cref="ComputeGraph"/>.</summary>
public sealed class Value
{
    public int Id { get; }
    public string Name { get; internal set; }
    public DataType Type { get; }
    public Shape Shape { get; }
    public ValueKind Kind { get; }
    public Node? Producer { get; internal set; }

    /// <summary>Set for <see cref="ValueKind.Weight"/> values.</summary>
    public WeightInfo? Weight { get; init; }

    /// <summary>Set for <see cref="ValueKind.State"/> values.</summary>
    public StateDefinition? State { get; init; }

    /// <summary>Set for <see cref="ValueKind.Input"/> values.</summary>
    public InputDefinition? Input { get; init; }

    /// <summary>Set for <see cref="ValueKind.Constant"/> values (raw little-endian bytes).</summary>
    public byte[]? ConstantData { get; init; }

    internal Value(int id, string name, DataType type, Shape shape, ValueKind kind)
    {
        Id = id;
        Name = name;
        Type = type;
        Shape = shape;
        Kind = kind;
    }

    public int Rank => Shape.Rank;

    public override string ToString() => $"%{Id}:{Name} {Type}{Shape}";
}
