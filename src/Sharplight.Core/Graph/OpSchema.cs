namespace Sharplight.Graph;

/// <summary>
/// Describes an operator: its name, category and shape/type inference.
/// Ops are registered in <see cref="OpRegistry"/>; model adapters may register their own ops and
/// backends provide kernels for whatever subset they support.
/// </summary>
public sealed class OpSchema
{
    /// <summary>Unique op name, used in the registry, in saved packages and by backends to select kernels.</summary>
    public string Name { get; }
    /// <summary>Broad classification used by fusion and placement.</summary>
    public OpCategory Category { get; }
    internal int MinInputs { get; }
    /// <summary>Largest number of inputs the op accepts (optional trailing inputs make this larger than the minimum).</summary>
    public int MaxInputs { get; }
    /// <summary>Human-readable description of the op's inputs, attributes and outputs.</summary>
    public string Description { get; }
    private readonly Func<IReadOnlyList<Value>, Attributes, IReadOnlyList<TensorType>> _infer;

    /// <summary>
    /// Creates an op description.
    /// </summary>
    /// <param name="name">Unique op name.</param>
    /// <param name="category">Broad classification of the op.</param>
    /// <param name="minInputs">Fewest inputs the op accepts.</param>
    /// <param name="maxInputs">Most inputs the op accepts.</param>
    /// <param name="infer">Computes the output types and shapes from the inputs and attributes; may throw for invalid operands.</param>
    /// <param name="description">Documentation of inputs, attributes and outputs.</param>
    public OpSchema(string name, OpCategory category, int minInputs, int maxInputs,
        Func<IReadOnlyList<Value>, Attributes, IReadOnlyList<TensorType>> infer, string description = "")
    {
        Name = name;
        Category = category;
        MinInputs = minInputs;
        MaxInputs = maxInputs;
        _infer = infer;
        Description = description;
    }

    internal IReadOnlyList<TensorType> Infer(IReadOnlyList<Value> inputs, Attributes attrs)
    {
        if (inputs.Count < MinInputs || inputs.Count > MaxInputs)
        {
            throw new ArgumentException($"{Name} expects {MinInputs}..{MaxInputs} inputs, got {inputs.Count}.");
        }

        return _infer(inputs, attrs);
    }

    public override string ToString() => Name;
}
