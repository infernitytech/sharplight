using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Package;

/// <summary>
/// Serializes a run of graph nodes (a placed <see cref="GraphSegment"/>, or one node) as a standalone graph, so that another process can
/// compile and run it on its own devices: the values the nodes read from outside become the graph's inputs (in order), the values later
/// segments read become its outputs, weights are referenced by name (the reader resolves them against its own copy of the model and checks
/// type and shape) and states travel whole. Like <see cref="DefinitionCodec"/> the nodes are replayed through <see cref="GraphBuilder"/>,
/// which runs every op's type inference again; a result that differs from the recorded output types fails the read.
/// </summary>
internal static class SegmentCodec
{
    private enum Rec : byte { Weight = 1, State, Constant, Node, End }

    /// <summary>The name of segment output <paramref name="index"/> in the graph <see cref="Read"/> returns: the parent graph's output name, if it is one.</summary>
    public static string OutputName(GraphSegment segment, int index) => OutputName(segment.Graph, segment.Outputs[index], index);

    /// <summary>Writes <paramref name="segment"/>: its nodes, inputs and outputs, and its graph's runtime scalars and data-dependent symbols.</summary>
    /// <exception cref="NotSupportedException">The segment holds a control-flow node or exchanges strings.</exception>
    public static void Write(BinaryWriter w, GraphSegment segment)
        => Write(w, $"{segment.Graph.Name}#{segment.Index}", segment.Graph, segment.Nodes, segment.Inputs, segment.Outputs);

    /// <summary>
    /// Writes one node as a graph whose inputs are its operands that are not leaves and whose outputs are its outputs (for questions about the
    /// node such as op support; the graph's runtime scalars and data-dependent symbols are left out).
    /// </summary>
    public static void WriteNode(BinaryWriter w, Node node)
    {
        var inputs = node.Inputs.Where(v => v.Kind is ValueKind.Input or ValueKind.Intermediate).Distinct().ToList();
        Write(w, $"node {node.Id}", graph: null, [node], inputs, node.Outputs);
    }

    private static void Write(BinaryWriter w, string name, ComputeGraph? graph, IReadOnlyList<Node> nodes, IReadOnlyList<Value> inputs, IReadOnlyList<Value> outputs)
    {
        MetadataCodec.WriteString(w, name);
        var index = new Dictionary<Value, int>();
        w.Write(inputs.Count);
        foreach (var v in inputs)
        {
            RejectStrings(v);
            MetadataCodec.WriteString(w, v.Name);
            w.Write((int)(v.Input?.Role ?? InputRole.Other));
            MetadataCodec.WriteString(w, v.Type.Name);
            MetadataCodec.WriteShape(w, v.Shape);
            index[v] = index.Count;
        }

        var scalars = graph?.RuntimeScalars ?? [];
        w.Write(scalars.Count);
        foreach (var s in scalars)
        {
            MetadataCodec.WriteString(w, s);
        }

        var dynamicSymbols = graph?.DynamicSymbols ?? [];
        w.Write(dynamicSymbols.Count);
        foreach (var (symbol, bound) in dynamicSymbols)
        {
            MetadataCodec.WriteString(w, symbol);
            MetadataCodec.WriteString(w, bound.ToString());
        }

        int Ref(Value v)
        {
            if (index.TryGetValue(v, out var i))
            {
                return i;
            }

            switch (v.Kind)
            {
                case ValueKind.Weight:
                    w.Write((byte)Rec.Weight);
                    MetadataCodec.WriteString(w, v.Weight!.Name);
                    MetadataCodec.WriteString(w, v.Type.Name);
                    MetadataCodec.WriteShape(w, v.Shape);
                    break;
                case ValueKind.State:
                    w.Write((byte)Rec.State);
                    WriteState(w, v.State!);
                    break;
                case ValueKind.Constant:
                    w.Write((byte)Rec.Constant);
                    MetadataCodec.WriteString(w, v.Name);
                    MetadataCodec.WriteString(w, v.Type.Name);
                    MetadataCodec.WriteShape(w, v.Shape);
                    w.Write(v.ConstantData!.Length);
                    w.Write(v.ConstantData);
                    break;
                default:
                    throw new InvalidDataException($"Segment {name}: value {v} is used before it is produced.");
            }

            return index[v] = index.Count;
        }

        foreach (var n in nodes)
        {
            if (n.Op.Category == OpCategory.ControlFlow || n.Attributes.Values.Any(a => a is ComputeGraph))
            {
                throw new NotSupportedException($"Segment {name}: control-flow node {n.Op.Name} (node {n.Id}) cannot run in another process.");
            }

            var ins = n.Inputs.Select(Ref).ToArray();
            var writes = n.Writes.Select(Ref).ToArray();
            w.Write((byte)Rec.Node);
            MetadataCodec.WriteString(w, n.Op.Name);
            MetadataCodec.WriteString(w, n.Group);
            MetadataCodec.Write(w, n.Attributes);
            WriteRefs(w, ins);
            WriteRefs(w, writes);
            string? baseName = DefinitionCodec.OutputBaseName(n);
            w.Write(baseName is not null);
            if (baseName is not null)
            {
                MetadataCodec.WriteString(w, baseName);
            }

            w.Write(n.Outputs.Length);
            foreach (var o in n.Outputs)
            {
                RejectStrings(o);
                MetadataCodec.WriteString(w, o.Type.Name);
                MetadataCodec.WriteShape(w, o.Shape);
                index[o] = index.Count;
            }
        }

        w.Write((byte)Rec.End);
        if (outputs.Count == 0)
        {
            // Nodes that only write state (a cache write): a graph needs an output, so the first operand stands in (never read).
            w.Write(1);
            MetadataCodec.WriteString(w, PlaceholderOutput);
            w.Write(Ref(nodes[0].Inputs.Concat(nodes[0].Writes).First()));
            return;
        }

        w.Write(outputs.Count);
        for (int i = 0; i < outputs.Count; i++)
        {
            MetadataCodec.WriteString(w, OutputName(graph, outputs[i], i));
            w.Write(Ref(outputs[i]));
        }
    }

    /// <summary>The output of a segment without outputs (see <see cref="Write(BinaryWriter, GraphSegment)"/>); it follows the segment's own outputs.</summary>
    public const string PlaceholderOutput = "%placeholder";

    /// <summary>
    /// Reads a segment written by <see cref="Write(BinaryWriter, GraphSegment)"/> or <see cref="WriteNode"/> as a graph: its inputs in the
    /// written order, its outputs named by <see cref="OutputName(GraphSegment, int)"/>.
    /// </summary>
    /// <param name="r">The reader.</param>
    /// <param name="weights">Resolves a weight name against the reader's copy of the model.</param>
    /// <param name="outputNames">The graph's output names in the segment's output order.</param>
    /// <exception cref="InvalidDataException">A weight is missing or differs in type or shape, an op is unknown, or type inference differs from the writer's.</exception>
    public static ComputeGraph Read(BinaryReader r, Func<string, WeightInfo?> weights, out IReadOnlyList<string> outputNames)
    {
        var name = MetadataCodec.ReadString(r);
        var b = new GraphBuilder(name, weights);
        var values = new List<Value>();
        int inputCount = r.ReadInt32();
        for (int i = 0; i < inputCount; i++)
        {
            var inputName = MetadataCodec.ReadString(r);
            var role = (InputRole)r.ReadInt32();
            var type = DataType.Get(MetadataCodec.ReadString(r));
            values.Add(b.Input(inputName, role, type, MetadataCodec.ReadShape(r)));
        }

        int scalars = r.ReadInt32();
        for (int i = 0; i < scalars; i++)
        {
            b.RuntimeScalar(MetadataCodec.ReadString(r));
        }

        int dynamic = r.ReadInt32();
        for (int i = 0; i < dynamic; i++)
        {
            var symbol = MetadataCodec.ReadString(r);
            b.DeclareDynamicSymbol(symbol, Dim.Sym(MetadataCodec.ReadString(r)));
        }

        for (var rec = (Rec)r.ReadByte(); rec != Rec.End; rec = (Rec)r.ReadByte())
        {
            switch (rec)
            {
                case Rec.Weight:
                    values.Add(ReadWeight(r, b, name));
                    break;
                case Rec.State:
                    values.Add(b.State(ReadState(r)));
                    break;
                case Rec.Constant:
                {
                    var constantName = MetadataCodec.ReadString(r);
                    var type = DataType.Get(MetadataCodec.ReadString(r));
                    var shape = MetadataCodec.ReadShape(r);
                    values.Add(b.Constant(constantName, type, shape, r.ReadBytes(r.ReadInt32())));
                    break;
                }
                case Rec.Node:
                    values.AddRange(ReadNode(r, b, values, name));
                    break;
                default:
                    throw new InvalidDataException($"Segment {name}: corrupt record {rec}.");
            }
        }

        int outputs = r.ReadInt32();
        var names = new string[outputs];
        for (int i = 0; i < outputs; i++)
        {
            names[i] = MetadataCodec.ReadString(r);
            b.Output(names[i], values[CheckedRef(r.ReadInt32(), values, name)]);
        }

        outputNames = names;
        return b.Build();
    }

    private static IReadOnlyList<Value> ReadNode(BinaryReader r, GraphBuilder b, List<Value> values, string segment)
    {
        var op = MetadataCodec.ReadString(r);
        var group = MetadataCodec.ReadString(r);
        var attrs = Attributes.From([.. MetadataCodec.Read(r).Select(kv => (kv.Key, kv.Value))]);
        var ins = ReadRefs(r, values, segment);
        var writes = ReadRefs(r, values, segment);
        string? baseName = r.ReadBoolean() ? MetadataCodec.ReadString(r) : null;
        int count = r.ReadInt32();
        var expected = new (DataType Type, Shape Shape)[count];
        for (int i = 0; i < count; i++)
        {
            expected[i] = (DataType.Get(MetadataCodec.ReadString(r)), MetadataCodec.ReadShape(r));
        }

        if (!OpRegistry.TryGet(op, out var schema))
        {
            throw new InvalidDataException($"Segment {segment} uses op '{op}', which this process does not know (a different Sharplight version?).");
        }

        IReadOnlyList<Value> outs;
        using (b.Group(group))
        {
            outs = b.Emit(schema, ins, attrs, baseName, writes.Length > 0 ? writes : null);
        }

        if (outs.Count != count || outs.Where((o, i) => o.Type != expected[i].Type || !o.Shape.Equals(expected[i].Shape)).Any())
        {
            throw new InvalidDataException($"Segment {segment}: op {op} infers {string.Join(", ", outs.Select(o => $"{o.Type}{o.Shape}"))} here, "
                + $"the sender recorded {string.Join(", ", expected.Select(e => $"{e.Type}{e.Shape}"))} (a different Sharplight version?).");
        }

        return outs;
    }

    private static Value ReadWeight(BinaryReader r, GraphBuilder b, string segment)
    {
        var weightName = MetadataCodec.ReadString(r);
        var type = DataType.Get(MetadataCodec.ReadString(r));
        var shape = MetadataCodec.ReadShape(r);
        var value = b.TryWeight(weightName) ?? throw new InvalidDataException($"Segment {segment} reads weight '{weightName}', which this copy of the model does not have.");
        if (value.Type != type || !value.Shape.Equals(shape))
        {
            throw new InvalidDataException($"Segment {segment}: weight '{weightName}' is {value.Type}{value.Shape} in this copy of the model, the sender's is {type}{shape}.");
        }

        return value;
    }

    private static void WriteState(BinaryWriter w, StateDefinition s)
    {
        MetadataCodec.WriteString(w, s.Name);
        MetadataCodec.WriteString(w, s.Type.Name);
        MetadataCodec.WriteShape(w, s.Shape);
        w.Write((int)s.Kind);
        w.Write(s.Compression);
        MetadataCodec.WriteString(w, s.Group);
    }

    private static StateDefinition ReadState(BinaryReader r) => new()
    {
        Name = MetadataCodec.ReadString(r),
        Type = DataType.Get(MetadataCodec.ReadString(r)),
        Shape = MetadataCodec.ReadShape(r),
        Kind = (StateKind)r.ReadInt32(),
        Compression = r.ReadInt32(),
        Group = MetadataCodec.ReadString(r),
    };

    private static string OutputName(ComputeGraph? graph, Value value, int index)
        => graph?.Outputs.FirstOrDefault(kv => kv.Value == value).Key ?? $"%{index}";

    private static void RejectStrings(Value v)
    {
        if (v.Type.Kind == DataTypeKind.String)
        {
            throw new NotSupportedException($"Value {v} holds strings, which stay in the process that owns them.");
        }
    }

    private static void WriteRefs(BinaryWriter w, int[] refs)
    {
        w.Write(refs.Length);
        foreach (var i in refs)
        {
            w.Write(i);
        }
    }

    private static Value[] ReadRefs(BinaryReader r, List<Value> values, string segment)
    {
        var a = new Value[r.ReadInt32()];
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = values[CheckedRef(r.ReadInt32(), values, segment)];
        }

        return a;
    }

    private static int CheckedRef(int index, List<Value> values, string segment)
        => index >= 0 && index < values.Count ? index : throw new InvalidDataException($"Segment {segment}: corrupt value reference {index}.");
}
