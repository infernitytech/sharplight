using System.Collections.Immutable;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Graph;

/// <summary>
/// Fluent builder used by model adapters to emit a <see cref="ComputeGraph"/>.
/// Weights and states are resolved lazily so the same builder can be shared by several graphs of a model.
/// </summary>
public sealed class GraphBuilder
{
    private readonly string _name;
    private readonly List<Node> _nodes = [];
    private readonly List<Value> _inputs = [];
    private readonly Dictionary<string, Value> _outputs = [];
    private readonly Dictionary<string, Value> _weightValues = [];
    private readonly Dictionary<string, Value> _stateValues = [];
    private readonly HashSet<string> _scalars = [];
    private readonly Func<string, WeightInfo?> _weightLookup;
    private readonly Stack<string> _groups = new();
    private int _nextValueId;
    private int _nextNodeId;

    /// <summary>
    /// Creates a builder for one graph.
    /// </summary>
    /// <param name="name">Graph name.</param>
    /// <param name="weightLookup">Resolves a weight name to its description, or <see langword="null"/> when the model has no such weight.</param>
    public GraphBuilder(string name, Func<string, WeightInfo?> weightLookup)
    {
        _name = name;
        _weightLookup = weightLookup;
    }

    private string CurrentGroup => _groups.Count > 0 ? _groups.Peek() : "";

    /// <summary>Opens a placement group scope (e.g. one transformer block).</summary>
    public IDisposable Group(string name)
    {
        _groups.Push(name);
        return new Scope(() => _groups.Pop());
    }

    // ---- leaves ---------------------------------------------------------------------------
    /// <summary>
    /// Declares a graph input the caller supplies for each execution.
    /// </summary>
    /// <param name="name">Input name, unique within the graph.</param>
    /// <param name="role">What the input means to the runtime (token ids, positions, ...).</param>
    /// <param name="type">Element type.</param>
    /// <param name="shape">Logical shape; the outermost dimension is usually symbolic.</param>
    public Value Input(string name, InputRole role, DataType type, Shape shape)
    {
        var v = new Value(_nextValueId++, name, type, shape, ValueKind.Input) { Input = new InputDefinition { Name = name, Role = role } };
        _inputs.Add(v);
        return v;
    }

    /// <summary>References a weight by name; throws if it does not exist.</summary>
    public Value Weight(string name) => TryWeight(name) ?? throw new KeyNotFoundException($"Missing weight '{name}'.");

    /// <summary>
    /// References a weight by name, creating the weight value on first use.
    /// </summary>
    /// <param name="name">Weight name as known to the weight lookup.</param>
    /// <returns>The weight value (the same instance for repeated calls), or <see langword="null"/> when the model has no such weight.</returns>
    public Value? TryWeight(string name)
    {
        if (_weightValues.TryGetValue(name, out var v))
        {
            return v;
        }

        var info = _weightLookup(name);
        if (info is null)
        {
            return null;
        }

        v = new Value(_nextValueId++, name, info.Type, info.Shape, ValueKind.Weight) { Weight = info };
        _weightValues[name] = v;
        return v;
    }

    /// <summary>
    /// Whether the model provides a weight with this name.
    /// </summary>
    /// <param name="name">Weight name.</param>
    public bool HasWeight(string name) => _weightValues.ContainsKey(name) || _weightLookup(name) is not null;

    /// <summary>
    /// References a persistent state (KV cache, recurrent state); repeated calls with the same name return the same value.
    /// </summary>
    /// <param name="def">The state description.</param>
    public Value State(StateDefinition def)
    {
        if (_stateValues.TryGetValue(def.Name, out var v))
        {
            return v;
        }

        v = new Value(_nextValueId++, def.Name, def.Type, def.Shape, ValueKind.State) { State = def };
        _stateValues[def.Name] = v;
        return v;
    }

    /// <summary>
    /// Embeds a small constant tensor in the graph.
    /// </summary>
    /// <param name="name">Display name.</param>
    /// <param name="type">Element type.</param>
    /// <param name="shape">Fully static shape.</param>
    /// <param name="data">Raw little-endian bytes; must be exactly the byte size of <paramref name="type"/> times the element count.</param>
    public Value Constant(string name, DataType type, Shape shape, byte[] data)
        => new(_nextValueId++, name, type, shape, ValueKind.Constant) { ConstantData = data };

    /// <summary>Declares a runtime scalar (e.g. <c>n_past</c>) used by ops of this graph.</summary>
    public void RuntimeScalar(string name) => _scalars.Add(name);

    /// <summary>
    /// Marks a value as a named graph output; a later call with the same name replaces the earlier one.
    /// </summary>
    /// <param name="name">Output name (for example <c>logits</c>).</param>
    /// <param name="v">The value to expose.</param>
    public void Output(string name, Value v) => _outputs[name] = v;

    // ---- generic op emission ----------------------------------------------------------------
    /// <summary>
    /// Emits an op looked up by name in <see cref="OpRegistry"/>.
    /// </summary>
    /// <param name="op">Op name.</param>
    /// <param name="inputs">Operands.</param>
    /// <param name="attrs">Op attributes, or <see langword="null"/> for none.</param>
    /// <param name="name">Base name for the output values, or <see langword="null"/> for generated names.</param>
    /// <param name="writes">States this node updates in place (side effects), or <see langword="null"/>.</param>
    /// <returns>The output values, typed by the op's shape inference.</returns>
    /// <exception cref="KeyNotFoundException">The op is not registered.</exception>
    /// <exception cref="ArgumentException">The number of inputs is outside the op's range, or shape inference rejected the operands.</exception>
    public IReadOnlyList<Value> Emit(string op, IReadOnlyList<Value> inputs, Attributes? attrs = null, string? name = null, IReadOnlyList<Value>? writes = null)
        => Emit(OpRegistry.Get(op), inputs, attrs, name, writes);

    /// <summary>
    /// Emits an op in the current placement group.
    /// </summary>
    /// <param name="op">Op schema.</param>
    /// <param name="inputs">Operands.</param>
    /// <param name="attrs">Op attributes, or <see langword="null"/> for none.</param>
    /// <param name="name">Base name for the output values (suffixed with the output index when the op has several), or <see langword="null"/> for generated names.</param>
    /// <param name="writes">States this node updates in place (side effects), or <see langword="null"/>.</param>
    /// <returns>The output values, typed by the op's shape inference.</returns>
    /// <exception cref="ArgumentException">The number of inputs is outside the op's range, or shape inference rejected the operands.</exception>
    public IReadOnlyList<Value> Emit(OpSchema op, IReadOnlyList<Value> inputs, Attributes? attrs = null, string? name = null, IReadOnlyList<Value>? writes = null)
    {
        attrs ??= Attributes.Empty;
        var types = op.Infer(inputs, attrs);
        var node = new Node(_nextNodeId++, op, [.. inputs], attrs, CurrentGroup) { Writes = writes is null ? [] : [.. writes] };
        var outs = ImmutableArray.CreateBuilder<Value>(types.Count);
        for (int i = 0; i < types.Count; i++)
        {
            var vname = name is null ? $"{op.Name.ToLowerInvariant()}_{node.Id}" : types.Count == 1 ? name : $"{name}.{i}";
            outs.Add(new Value(_nextValueId++, vname, types[i].Type, types[i].Shape, ValueKind.Intermediate));
        }
        node.Outputs = outs.MoveToImmutable();
        foreach (var o in node.Outputs)
        {
            o.Producer = node;
        }

        _nodes.Add(node);
        return node.Outputs;
    }

    /// <summary>
    /// Emits a single-output op and returns its output.
    /// </summary>
    /// <param name="op">Op name.</param>
    /// <param name="inputs">Operands.</param>
    /// <param name="attrs">Op attributes, or <see langword="null"/> for none.</param>
    /// <param name="name">Name of the output value, or <see langword="null"/> for a generated name.</param>
    public Value Emit1(string op, IReadOnlyList<Value> inputs, Attributes? attrs = null, string? name = null)
        => Emit(op, inputs, attrs, name)[0];

    // ---- typed helpers ------------------------------------------------------------------------
    /// <summary>Emits a one-operand op such as <c>Silu</c>.</summary>
    public Value Unary(string op, Value x, string? name = null) => Emit1(op, [x], null, name);
    /// <summary>Emits a two-operand broadcasting op such as <c>Add</c>.</summary>
    public Value Binary(string op, Value a, Value b, string? name = null) => Emit1(op, [a, b], null, name);
    /// <summary>Elementwise sum with broadcasting.</summary>
    public Value Add(Value a, Value b, string? name = null) => Binary("Add", a, b, name);
    /// <summary>Elementwise product with broadcasting.</summary>
    public Value Mul(Value a, Value b, string? name = null) => Binary("Mul", a, b, name);
    /// <summary>Elementwise difference with broadcasting.</summary>
    public Value Sub(Value a, Value b, string? name = null) => Binary("Sub", a, b, name);
    /// <summary>SiLU activation, <c>x * sigmoid(x)</c>.</summary>
    public Value Silu(Value x, string? name = null) => Unary("Silu", x, name);
    /// <summary>Logistic sigmoid.</summary>
    public Value Sigmoid(Value x, string? name = null) => Unary("Sigmoid", x, name);
    /// <summary>Softplus, <c>log(1 + exp(x))</c>.</summary>
    public Value Softplus(Value x, string? name = null) => Unary("Softplus", x, name);
    /// <summary>Elementwise <c>x * scale + bias</c>.</summary>
    public Value Scale(Value x, float scale, float bias = 0, string? name = null)
        => Emit1("Scale", [x], Attributes.From(("scale", scale), ("bias", bias)), name);

    /// <summary>Matrix product of <paramref name="x"/> [..., K] with a weight laid out [N, K], giving [..., N].</summary>
    public Value MatMul(Value x, Value w, string? name = null) => Emit1("MatMul", [x, w], null, name);

    /// <summary>Matrix product with the named weight, plus an optional bias weight that is added when the model has it.</summary>
    public Value Linear(Value x, string weight, string? biasWeight = null, string? name = null)
    {
        var y = MatMul(x, Weight(weight), name);
        if (biasWeight is not null && TryWeight(biasWeight) is { } b)
        {
            y = Add(y, b);
        }

        return y;
    }

    /// <summary>
    /// RMS normalization over the last axis, optionally multiplied by a learned weight.
    /// </summary>
    /// <param name="x">Input.</param>
    /// <param name="weight">Per-channel scale, or <see langword="null"/> for none.</param>
    /// <param name="eps">Value added to the mean square for numerical stability.</param>
    /// <param name="name">Output name, or <see langword="null"/>.</param>
    public Value RmsNorm(Value x, Value? weight, float eps, string? name = null)
        => Emit1("RmsNorm", weight is null ? [x] : [x, weight], Attributes.From(("eps", eps)), name);

    /// <summary>Divides by the L2 norm over the last axis.</summary>
    public Value L2Norm(Value x, float eps, string? name = null) => Emit1("L2Norm", [x], Attributes.From(("eps", eps)), name);

    /// <summary>Reinterprets the value with another shape (element count unchanged); one entry may be -1 to infer it.</summary>
    public Value Reshape(Value x, params long[] shape) => Emit1("Reshape", [x], Attributes.From(("shape", shape)));

    /// <summary>
    /// Takes <paramref name="length"/> entries starting at <paramref name="start"/> along <paramref name="axis"/>.
    /// </summary>
    public Value Slice(Value x, int axis, long start, long length, string? name = null)
        => Emit1("Slice", [x], Attributes.From(("axis", axis), ("start", start), ("length", length)), name);

    /// <summary>Reorders the axes: output axis <c>i</c> is input axis <c>perm[i]</c>.</summary>
    public Value Permute(Value x, params int[] perm) => Emit1("Permute", [x], Attributes.From(("perm", perm)));
    /// <summary>Materializes a view (for example after <see cref="Permute"/>) into a dense row-major tensor.</summary>
    public Value Contiguous(Value x) => Emit1("Contiguous", [x]);
    /// <summary>Concatenates the operands along <paramref name="axis"/>.</summary>
    public Value Concat(int axis, params Value[] xs) => Emit1("Concat", xs, Attributes.From(("axis", axis)));
    /// <summary>Selects rows of <paramref name="table"/> by integer <paramref name="indices"/> (embedding lookup).</summary>
    public Value Gather(Value table, Value indices, string? name = null) => Emit1("Gather", [table, indices], null, name);

    /// <summary>
    /// Appends <paramref name="x"/> to a KV-cache state at the row given by the runtime scalar <paramref name="offsetScalar"/>.
    /// </summary>
    /// <param name="state">The cache state (written in place).</param>
    /// <param name="x">The new rows.</param>
    /// <param name="offsetScalar">Name of the runtime scalar holding the first row to write; declared on the graph automatically.</param>
    public void CacheWrite(Value state, Value x, string offsetScalar = WellKnown.PastScalar)
    {
        RuntimeScalar(offsetScalar);
        Emit("CacheWrite", [state, x], Attributes.From(("offset", offsetScalar)), writes: [state]);
    }

    /// <summary>
    /// Causal attention of <paramref name="q"/> against the first rows of the K and V caches, taking positions from <paramref name="offsetScalar"/>.
    /// </summary>
    /// <param name="q">Queries.</param>
    /// <param name="kCache">Key cache state.</param>
    /// <param name="vCache">Value cache state.</param>
    /// <param name="scale">Score scale, usually <c>1/sqrt(headDim)</c>.</param>
    /// <param name="offsetScalar">Name of the runtime scalar holding the number of cached tokens before this call.</param>
    /// <param name="name">Output name, or <see langword="null"/>.</param>
    public Value CachedAttention(Value q, Value kCache, Value vCache, float scale, string offsetScalar = WellKnown.PastScalar, string? name = null)
    {
        RuntimeScalar(offsetScalar);
        return Emit1("CachedAttention", [q, kCache, vCache], Attributes.From(("scale", scale), ("offset", offsetScalar)), name);
    }

    /// <summary>
    /// Finishes the graph.
    /// </summary>
    /// <returns>The immutable graph.</returns>
    /// <exception cref="InvalidOperationException">No output was declared.</exception>
    public ComputeGraph Build()
    {
        if (_outputs.Count == 0)
        {
            throw new InvalidOperationException("Graph has no outputs.");
        }

        return new ComputeGraph(_name, [.. _nodes], [.. _inputs], new Dictionary<string, Value>(_outputs), [.. _scalars]);
    }

    private sealed class Scope(Action onDispose) : IDisposable
    {
        public void Dispose() => onDispose();
    }
}
