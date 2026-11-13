namespace Sharplight.Graph;

/// <summary>
/// Describes an operator: its name, category and shape/type inference.
/// Ops are registered in <see cref="OpRegistry"/>; model adapters may register their own ops and
/// backends provide kernels for whatever subset they support.
/// </summary>
public sealed class OpSchema
{
    public string Name { get; }
    public OpCategory Category { get; }
    public int MinInputs { get; }
    public int MaxInputs { get; }
    public string Description { get; }
    private readonly Func<IReadOnlyList<Value>, Attributes, IReadOnlyList<TensorType>> _infer;

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

    public IReadOnlyList<TensorType> Infer(IReadOnlyList<Value> inputs, Attributes attrs)
    {
        if (inputs.Count < MinInputs || inputs.Count > MaxInputs)
        {
            throw new ArgumentException($"{Name} expects {MinInputs}..{MaxInputs} inputs, got {inputs.Count}.");
        }

        return _infer(inputs, attrs);
    }

    public override string ToString() => Name;
}
