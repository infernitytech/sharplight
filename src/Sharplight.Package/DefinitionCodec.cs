using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Package;

/// <summary>
/// Serializes a built <see cref="ModelDefinition"/> (Sharplight IR: graphs, states, properties, source metadata).
/// Graphs are stored as a replayable op list and rebuilt through <see cref="GraphBuilder"/>, which re-runs every
/// op's type inference — a stored graph is validated against the current op definitions when loaded.
/// Weights are referenced by name.
/// </summary>
internal static class DefinitionCodec
{
    private enum Rec : byte { Input = 1, Weight, State, Constant, Node, End }

    public static void Write(BinaryWriter w, ModelDefinition def)
    {
        MetadataCodec.WriteString(w, def.Architecture);
        MetadataCodec.WriteString(w, def.Name);
        MetadataCodec.Write(w, def.Properties);
        MetadataCodec.Write(w, def.SourceMetadata);
        w.Write(def.States.Count);
        foreach (var s in def.States)
        {
            WriteState(w, s);
        }

        w.Write(def.Graphs.Count);
        foreach (var (key, g) in def.Graphs.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            MetadataCodec.WriteString(w, key);
            WriteGraph(w, g, def);
        }
    }

    public static ModelDefinition Read(BinaryReader r, IReadOnlyDictionary<string, WeightInfo> weights, IReadOnlyList<IDisposable> owned)
    {
        string arch = MetadataCodec.ReadString(r);
        string name = MetadataCodec.ReadString(r);
        var props = MetadataCodec.Read(r);
        var meta = MetadataCodec.Read(r);
        int ns = r.ReadInt32();
        var states = new List<StateDefinition>(ns);
        for (int i = 0; i < ns; i++)
        {
            states.Add(ReadState(r));
        }

        var stateByName = states.ToDictionary(s => s.Name, StringComparer.Ordinal);
        int ng = r.ReadInt32();
        var graphs = new Dictionary<string, ComputeGraph>(ng, StringComparer.Ordinal);
        for (int i = 0; i < ng; i++)
        {
            var key = MetadataCodec.ReadString(r);
            graphs[key] = ReadGraph(r, weights, stateByName);
        }
        return new ModelDefinition
        {
            Architecture = arch,
            Name = name,
            Weights = weights,
            Graphs = graphs,
            States = states,
            Properties = props,
            SourceMetadata = meta,
            OwnedResources = owned,
        };
    }

    private static void WriteState(BinaryWriter w, StateDefinition s)
    {
        MetadataCodec.WriteString(w, s.Name);
        MetadataCodec.WriteString(w, s.Type.Name);
        MetadataCodec.WriteShape(w, s.Shape);
        w.Write((int)s.Kind);
        MetadataCodec.WriteString(w, s.Group);
    }

    private static StateDefinition ReadState(BinaryReader r) => new()
    {
        Name = MetadataCodec.ReadString(r),
        Type = DataType.Get(MetadataCodec.ReadString(r)),
        Shape = MetadataCodec.ReadShape(r),
        Kind = (StateKind)r.ReadInt32(),
        Group = MetadataCodec.ReadString(r),
    };

    private static void WriteGraph(BinaryWriter w, ComputeGraph g, ModelDefinition def)
    {
        MetadataCodec.WriteString(w, g.Name);
        var index = new Dictionary<Value, int>();
        int next = 0;

        foreach (var v in g.Inputs)
        {
            w.Write((byte)Rec.Input);
            MetadataCodec.WriteString(w, v.Name);
            w.Write((int)v.Input!.Role);
            MetadataCodec.WriteString(w, v.Type.Name);
            MetadataCodec.WriteShape(w, v.Shape);
            index[v] = next++;
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
                    break;
                case ValueKind.State:
                    if (!def.States.Contains(v.State!))
                    {
                        throw new InvalidDataException($"Graph {g.Name} uses state {v.State!.Name}, which the model definition does not declare.");
                    }

                    w.Write((byte)Rec.State);
                    MetadataCodec.WriteString(w, v.State!.Name);
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
                    throw new InvalidDataException($"Graph {g.Name}: value {v} is used before it is produced.");
            }
            return index[v] = next++;
        }

        foreach (var n in g.Nodes)
        {
            var ins = n.Inputs.Select(Ref).ToArray();
            var writes = n.Writes.Select(Ref).ToArray();
            w.Write((byte)Rec.Node);
            MetadataCodec.WriteString(w, n.Op.Name);
            MetadataCodec.WriteString(w, n.Group);
            MetadataCodec.Write(w, n.Attributes);
            w.Write(ins.Length);
            foreach (var i in ins)
            {
                w.Write(i);
            }

            w.Write(writes.Length);
            foreach (var i in writes)
            {
                w.Write(i);
            }
            // Output names: single outputs keep theirs, multi-output nodes keep the common "{name}.{i}" prefix.
            string? name = n.Outputs.Length == 0 ? null : n.Outputs.Length == 1 ? n.Outputs[0].Name
                : n.Outputs[0].Name.EndsWith(".0", StringComparison.Ordinal) ? n.Outputs[0].Name[..^2] : null;
            w.Write(name is not null);
            if (name is not null)
            {
                MetadataCodec.WriteString(w, name);
            }

            w.Write(n.Outputs.Length);
            foreach (var o in n.Outputs)
            {
                index[o] = next++;
            }
        }
        foreach (var v in g.Outputs.Values)
        {
            Ref(v); // outputs that are leaves get their records before End
        }

        w.Write((byte)Rec.End);

        w.Write(g.Outputs.Count);
        foreach (var (k, v) in g.Outputs.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            MetadataCodec.WriteString(w, k);
            w.Write(Ref(v));
        }
        w.Write(g.RuntimeScalars.Count);
        foreach (var s in g.RuntimeScalars)
        {
            MetadataCodec.WriteString(w, s);
        }
    }

    private static ComputeGraph ReadGraph(BinaryReader r, IReadOnlyDictionary<string, WeightInfo> weights, Dictionary<string, StateDefinition> states)
    {
        var b = new GraphBuilder(MetadataCodec.ReadString(r), n => weights.GetValueOrDefault(n));
        var values = new List<Value>();
        for (var rec = (Rec)r.ReadByte(); rec != Rec.End; rec = (Rec)r.ReadByte())
        {
            switch (rec)
            {
                case Rec.Input:
                {
                    var name = MetadataCodec.ReadString(r);
                    var role = (InputRole)r.ReadInt32();
                    var type = DataType.Get(MetadataCodec.ReadString(r));
                    values.Add(b.Input(name, role, type, MetadataCodec.ReadShape(r)));
                    break;
                }
                case Rec.Weight:
                {
                    var name = MetadataCodec.ReadString(r);
                    values.Add(b.TryWeight(name) ?? throw new InvalidDataException($"Package graph references missing weight '{name}'."));
                    break;
                }
                case Rec.State:
                {
                    var name = MetadataCodec.ReadString(r);
                    values.Add(b.State(states.TryGetValue(name, out var s) ? s : throw new InvalidDataException($"Package graph references undeclared state '{name}'.")));
                    break;
                }
                case Rec.Constant:
                {
                    var name = MetadataCodec.ReadString(r);
                    var type = DataType.Get(MetadataCodec.ReadString(r));
                    var shape = MetadataCodec.ReadShape(r);
                    values.Add(b.Constant(name, type, shape, r.ReadBytes(r.ReadInt32())));
                    break;
                }
                case Rec.Node:
                {
                    var op = MetadataCodec.ReadString(r);
                    var group = MetadataCodec.ReadString(r);
                    var attrs = Attributes.From([.. MetadataCodec.Read(r).Select(kv => (kv.Key, kv.Value))]);
                    var ins = ReadRefs(r, values);
                    var writes = ReadRefs(r, values);
                    string? name = r.ReadBoolean() ? MetadataCodec.ReadString(r) : null;
                    int nOut = r.ReadInt32();
                    if (!OpRegistry.TryGet(op, out var schema))
                        {
                            throw new InvalidDataException($"Package graph uses unknown op '{op}'.");
                        }

                        IReadOnlyList<Value> outs;
                    using (b.Group(group))
                        {
                            outs = b.Emit(schema, ins, attrs, name, writes.Length > 0 ? writes : null);
                        }

                        if (outs.Count != nOut)
                        {
                            throw new InvalidDataException($"Op {op} now produces {outs.Count} outputs, the package stored {nOut}.");
                        }

                        values.AddRange(outs);
                    break;
                }
                default:
                    throw new InvalidDataException($"Corrupt package graph record {rec}.");
            }
        }
        int no = r.ReadInt32();
        for (int i = 0; i < no; i++)
        {
            var k = MetadataCodec.ReadString(r);
            b.Output(k, values[r.ReadInt32()]);
        }
        int nsc = r.ReadInt32();
        for (int i = 0; i < nsc; i++)
        {
            b.RuntimeScalar(MetadataCodec.ReadString(r));
        }

        return b.Build();
    }

    private static Value[] ReadRefs(BinaryReader r, List<Value> values)
    {
        var a = new Value[r.ReadInt32()];
        for (int i = 0; i < a.Length; i++)
        {
            int idx = r.ReadInt32();
            a[i] = idx >= 0 && idx < values.Count ? values[idx] : throw new InvalidDataException("Corrupt package graph value reference.");
        }
        return a;
    }
}
