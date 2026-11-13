using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Graph;

/// <summary>An SSA value (tensor) in a <see cref="ComputeGraph"/>.</summary>
public sealed class Value
{
    /// <summary>Unique id of the value within its graph (printed as <c>%Id</c> in dumps).</summary>
    public int Id { get; }
    /// <summary>Display name (weight/input/state name, or a generated name for intermediates).</summary>
    public string Name { get; internal set; }
    /// <summary>Element storage type.</summary>
    public DataType Type { get; }
    /// <summary>Logical shape, outermost dimension first. Only the outermost dimension may be symbolic.</summary>
    public Shape Shape { get; }
    /// <summary>How the value comes into existence.</summary>
    public ValueKind Kind { get; }
    /// <summary>The node that produces this value; <see langword="null"/> for inputs, weights, states and constants.</summary>
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

    /// <summary>Number of dimensions.</summary>
    public int Rank => Shape.Rank;

    /// <inheritdoc/>
    public override string ToString() => $"%{Id}:{Name} {Type}{Shape}";
}
