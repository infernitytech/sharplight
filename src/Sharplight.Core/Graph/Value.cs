using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Graph;

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
