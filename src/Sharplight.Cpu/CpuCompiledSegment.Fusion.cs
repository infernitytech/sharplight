using Sharplight.Graph;

namespace Sharplight.Cpu;

// Fusion of elementwise consumers into the op producing their input (conv epilogues).
internal sealed unsafe partial class CpuCompiledSegment
{
    /// <summary>Unary ops a conv epilogue applies per element (all of them elementwise and parameterized like <see cref="UnaryOp"/>).</summary>
    private static readonly HashSet<UnaryKind> s_epilogueActivations =
    [
        UnaryKind.Relu, UnaryKind.Clip, UnaryKind.LeakyRelu, UnaryKind.Silu, UnaryKind.Sigmoid, UnaryKind.HardSwish,
        UnaryKind.HardSigmoid, UnaryKind.Gelu, UnaryKind.GeluTanh, UnaryKind.Tanh, UnaryKind.Elu, UnaryKind.Mish, UnaryKind.Softplus,
    ];

    /// <summary><c>SHARPLIGHT_CPU_NO_FUSION=1</c> lowers every node on its own (no conv epilogues).</summary>
    private static readonly bool s_noFusion = Environment.GetEnvironmentVariable("SHARPLIGHT_CPU_NO_FUSION") == "1";

    /// <summary>Consumers of every value inside this segment.</summary>
    private readonly Dictionary<Value, List<Node>> _consumers = [];

    /// <summary>Nodes computed by the epilogue of an earlier op; lowering skips them.</summary>
    private readonly HashSet<Node> _absorbed = [];

    private void IndexConsumers()
    {
        foreach (var node in _segment.Nodes)
        {
            foreach (var input in node.Inputs.Distinct())
            {
                if (!_consumers.TryGetValue(input, out var list))
                {
                    _consumers[input] = list = [];
                }

                list.Add(node);
            }
        }
    }

    /// <summary>The only consumer of <paramref name="v"/> when it has exactly one (by one input) and is not a segment output.</summary>
    private Node? SoleConsumer(Value v)
    {
        if (s_noFusion || _segment.Outputs.Contains(v) || !_consumers.TryGetValue(v, out var list) || list.Count != 1)
        {
            return null;
        }

        var consumer = list[0];
        return consumer.Inputs.Count(i => i == v) == 1 ? consumer : null;
    }

    /// <summary>
    /// Absorbs the elementwise tail of an op producing <paramref name="y"/> (value <paramref name="output"/>): with
    /// <paramref name="columnBias"/>, an Add of a constant vector over the last axis; then an Add of a same-shaped, already
    /// computed dense tensor; then one unary activation, each the sole consumer of the previous value. The absorbed nodes'
    /// outputs alias <paramref name="y"/>. Returns null when nothing can be fused.
    /// </summary>
    private OutputEpilogue? AbsorbEpilogue(Value output, TRef y, bool columnBias = false)
    {
        TRef? biasVector = null;
        long columns = output.Shape.Rank > 0 && !output.Shape[-1].IsSymbolic ? output.Shape[-1].Fixed : 0;
        var current = output;
        var absorbed = new List<Node>();
        if (columnBias && columns > 0 && SoleConsumer(current) is { Op.Name: "Add" } biasAdd)
        {
            var other = biasAdd.Inputs[0] == current ? biasAdd.Inputs[1] : biasAdd.Inputs[0];
            if (IsStatic(other) && other.Type == Tensors.DataType.F32 && other.Shape.Rank <= output.Shape.Rank
                && other.Shape.Dims.All(d => !d.IsSymbolic) && other.Shape.Dims.Aggregate(1L, (acc, d) => acc * d.Fixed) == columns
                && other.Shape[-1].Fixed == columns && biasAdd.Output.Shape.Equals(output.Shape) && Ref(other) is { IsDense: true } b)
            {
                biasVector = b;
                absorbed.Add(biasAdd);
                current = biasAdd.Output;
            }
        }

        TRef? residual = null;
        if (SoleConsumer(current) is { Op.Name: "Add" } add)
        {
            var other = add.Inputs[0] == current ? add.Inputs[1] : add.Inputs[0];
            if (_refs.TryGetValue(other, out var r) && r.Type == Tensors.DataType.F32 && r.IsDense && r.Sym == y.Sym
                && r.Dims.AsSpan().SequenceEqual(y.Dims) && add.Output.Shape.Equals(output.Shape))
            {
                residual = r;
                absorbed.Add(add);
                current = add.Output;
            }
        }

        UnaryKind? activation = null;
        float scale = 1;
        float bias = 0;
        if (SoleConsumer(current) is { } act && ActivationKind(act.Op.Name) is { } kind && act.Output.Shape.Equals(output.Shape))
        {
            activation = kind;
            (scale, bias) = UnaryParameters(kind, act.Attributes);
            absorbed.Add(act);
        }
        else if (SiluPattern(current) is { } silu)
        {
            activation = UnaryKind.Silu;
            absorbed.Add(silu.Sigmoid);
            absorbed.Add(silu.Mul);
        }

        if (absorbed.Count == 0)
        {
            return null;
        }

        foreach (var node in absorbed)
        {
            _absorbed.Add(node);
            _refs[node.Output] = y;
        }

        return new OutputEpilogue
        {
            ColumnBias = biasVector, Columns = Math.Max(1, columns), Residual = residual, Activation = activation, Scale = scale, Bias = bias,
        };
    }

    /// <summary>
    /// SiLU written out as <c>Mul(v, Sigmoid(v))</c> (common in exported YOLO / EfficientNet graphs): v has exactly these two
    /// consumers and the sigmoid only feeds the product.
    /// </summary>
    private (Node Sigmoid, Node Mul)? SiluPattern(Value v)
    {
        if (s_noFusion || _segment.Outputs.Contains(v) || !_consumers.TryGetValue(v, out var list) || list.Count != 2)
        {
            return null;
        }

        var sigmoid = list.FirstOrDefault(c => c.Op.Name == "Sigmoid");
        var mul = list.FirstOrDefault(c => c.Op.Name == "Mul");
        if (sigmoid is null || mul is null || SoleConsumer(sigmoid.Output) != mul || !mul.Output.Shape.Equals(v.Shape))
        {
            return null;
        }

        bool operands = mul.Inputs[0] == v && mul.Inputs[1] == sigmoid.Output || mul.Inputs[1] == v && mul.Inputs[0] == sigmoid.Output;
        return operands ? (sigmoid, mul) : null;
    }

    private static UnaryKind? ActivationKind(string op)
        => op == "Clip" ? UnaryKind.Clip : s_unaryKinds.TryGetValue(op, out var k) && s_epilogueActivations.Contains(k) ? k : null;
}
