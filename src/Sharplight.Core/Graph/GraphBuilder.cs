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

    public GraphBuilder(string name, Func<string, WeightInfo?> weightLookup)
    {
        _name = name;
        _weightLookup = weightLookup;
    }

    public string CurrentGroup => _groups.Count > 0 ? _groups.Peek() : "";

    /// <summary>Opens a placement group scope (e.g. one transformer block).</summary>
    public IDisposable Group(string name)
    {
        _groups.Push(name);
        return new Scope(() => _groups.Pop());
    }

    // ---- leaves ---------------------------------------------------------------------------
    public Value Input(string name, InputRole role, DataType type, Shape shape)
    {
        var v = new Value(_nextValueId++, name, type, shape, ValueKind.Input) { Input = new InputDefinition { Name = name, Role = role } };
        _inputs.Add(v);
        return v;
    }

    /// <summary>References a weight by name; throws if it does not exist.</summary>
    public Value Weight(string name) => TryWeight(name) ?? throw new KeyNotFoundException($"Missing weight '{name}'.");

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

    public bool HasWeight(string name) => _weightValues.ContainsKey(name) || _weightLookup(name) is not null;

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

    public Value Constant(string name, DataType type, Shape shape, byte[] data)
        => new(_nextValueId++, name, type, shape, ValueKind.Constant) { ConstantData = data };

    /// <summary>Declares a runtime scalar (e.g. <c>n_past</c>) used by ops of this graph.</summary>
    public void RuntimeScalar(string name) => _scalars.Add(name);

    public void Output(string name, Value v) => _outputs[name] = v;

    // ---- generic op emission ----------------------------------------------------------------
    public IReadOnlyList<Value> Emit(string op, IReadOnlyList<Value> inputs, Attributes? attrs = null, string? name = null, IReadOnlyList<Value>? writes = null)
        => Emit(OpRegistry.Get(op), inputs, attrs, name, writes);

    public IReadOnlyList<Value> Emit(OpSchema op, IReadOnlyList<Value> inputs, Attributes? attrs = null, string? name = null, IReadOnlyList<Value>? writes = null)
    {
        attrs ??= Attributes.Empty;
        var types = op.Infer(inputs, attrs);
        var node = new Node(_nextNodeId++, op, [.. inputs], attrs, CurrentGroup) { Writes = writes is null ? [] : [.. writes] };
        var outs = ImmutableArray.CreateBuilder<Value>(types.Count);
        for (int i = 0; i < types.Count; i++)
        {
            var vname = name is null ? $"{op.Name.ToLowerInvariant()}_{node.Id}" : types.Count == 1 ? name : $"{name}.{i}";
            outs.Add(new Value(_nextValueId++, vname, types[i].Type, types[i].Shape, ValueKind.Intermediate) { });
        }
        node.Outputs = outs.MoveToImmutable();
        foreach (var o in node.Outputs)
        {
            o.Producer = node;
        }

        _nodes.Add(node);
        return node.Outputs;
    }

    public Value Emit1(string op, IReadOnlyList<Value> inputs, Attributes? attrs = null, string? name = null)
        => Emit(op, inputs, attrs, name)[0];

    // ---- typed helpers ------------------------------------------------------------------------
    public Value Unary(string op, Value x, string? name = null) => Emit1(op, [x], null, name);
    public Value Binary(string op, Value a, Value b, string? name = null) => Emit1(op, [a, b], null, name);
    public Value Add(Value a, Value b, string? name = null) => Binary("Add", a, b, name);
    public Value Mul(Value a, Value b, string? name = null) => Binary("Mul", a, b, name);
    public Value Sub(Value a, Value b, string? name = null) => Binary("Sub", a, b, name);
    public Value Silu(Value x, string? name = null) => Unary("Silu", x, name);
    public Value Sigmoid(Value x, string? name = null) => Unary("Sigmoid", x, name);
    public Value Softplus(Value x, string? name = null) => Unary("Softplus", x, name);
    public Value Scale(Value x, float scale, float bias = 0, string? name = null)
        => Emit1("Scale", [x], Attributes.From(("scale", scale), ("bias", bias)), name);

    public Value MatMul(Value x, Value w, string? name = null) => Emit1("MatMul", [x, w], null, name);

    /// <summary>MatMul by weight name; returns null if the weight does not exist (optional projections).</summary>
    public Value Linear(Value x, string weight, string? biasWeight = null, string? name = null)
    {
        var y = MatMul(x, Weight(weight), name);
        if (biasWeight is not null && TryWeight(biasWeight) is { } b)
        {
            y = Add(y, b);
        }

        return y;
    }

    public Value RmsNorm(Value x, Value? weight, float eps, string? name = null)
        => Emit1("RmsNorm", weight is null ? [x] : [x, weight], Attributes.From(("eps", eps)), name);

    public Value L2Norm(Value x, float eps, string? name = null) => Emit1("L2Norm", [x], Attributes.From(("eps", eps)), name);

    public Value Reshape(Value x, params long[] shape) => Emit1("Reshape", [x], Attributes.From(("shape", shape)));

    public Value Slice(Value x, int axis, long start, long length, string? name = null)
        => Emit1("Slice", [x], Attributes.From(("axis", axis), ("start", start), ("length", length)), name);

    public Value Permute(Value x, params int[] perm) => Emit1("Permute", [x], Attributes.From(("perm", perm)));
    public Value Contiguous(Value x) => Emit1("Contiguous", [x]);
    public Value Concat(int axis, params Value[] xs) => Emit1("Concat", xs, Attributes.From(("axis", axis)));
    public Value Gather(Value table, Value indices, string? name = null) => Emit1("Gather", [table, indices], null, name);

    public void CacheWrite(Value state, Value x, string offsetScalar = WellKnown.PastScalar)
    {
        RuntimeScalar(offsetScalar);
        Emit("CacheWrite", [state, x], Attributes.From(("offset", offsetScalar)), writes: [state]);
    }

    public Value CachedAttention(Value q, Value kCache, Value vCache, float scale, string offsetScalar = WellKnown.PastScalar, string? name = null)
    {
        RuntimeScalar(offsetScalar);
        return Emit1("CachedAttention", [q, kCache, vCache], Attributes.From(("scale", scale), ("offset", offsetScalar)), name);
    }

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
