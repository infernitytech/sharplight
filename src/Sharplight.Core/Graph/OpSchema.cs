using System.Collections.Concurrent;
using Sharplight.Tensors;

namespace Sharplight.Graph;

/// <summary>Broad classification used by optimization passes (e.g. elementwise fusion) and planners.</summary>
public enum OpCategory
{
    /// <summary>Pointwise op with numpy-style broadcasting; freely fusable.</summary>
    Elementwise,
    /// <summary>Re-interprets the input buffer (reshape, slice, permute): no computation, aliases its input.</summary>
    View,
    /// <summary>Reduction / normalization over the innermost axis.</summary>
    Reduction,
    /// <summary>Matrix multiplication family.</summary>
    MatMul,
    /// <summary>Data movement (gather, concat, copy, cache writes).</summary>
    Data,
    /// <summary>Higher-level fused operator (attention, rope, recurrent rules, ...).</summary>
    Composite,
}

/// <summary>Output type of an op as computed by shape inference.</summary>
public readonly record struct TensorType(DataType Type, Shape Shape);

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

/// <summary>Global, thread-safe registry of operator schemas.</summary>
public static class OpRegistry
{
    private static readonly ConcurrentDictionary<string, OpSchema> s_ops = new(StringComparer.Ordinal);

    static OpRegistry() => StandardOps.RegisterAll();

    public static OpSchema Register(OpSchema schema)
    {
        if (!s_ops.TryAdd(schema.Name, schema) && !ReferenceEquals(s_ops[schema.Name], schema))
        {
            throw new InvalidOperationException($"Op '{schema.Name}' is already registered.");
        }

        return schema;
    }

    public static OpSchema Get(string name) =>
        s_ops.TryGetValue(name, out var s) ? s : throw new KeyNotFoundException($"Unknown op '{name}'.");

    public static bool TryGet(string name, out OpSchema schema) => s_ops.TryGetValue(name, out schema!);

    public static IEnumerable<OpSchema> All => s_ops.Values;
}
