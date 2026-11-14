using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Sequences and optionals (see docs/dynamic-shapes.md, "Sequences and optionals"). They exist only in the translator: a sequence whose
// structure is known at import is a list of translated tensors, and every sequence op is resolved into ordinary IR on its elements. A
// sequence whose length is only known at run time (built by a runtime Loop) is a stacked tensor (TranslatedTensor.Stacked). An optional
// is its value, or TranslatedTensor.None when it is empty.
internal sealed partial class OnnxTranslator
{
    private static readonly HashSet<string> s_sequenceOps =
    [
        "SequenceEmpty", "SequenceConstruct", "SequenceInsert", "SequenceErase", "SequenceAt", "SequenceLength", "SplitToSequence",
        "ConcatFromSequence", "SequenceMap", "Optional", "OptionalHasElement", "OptionalGetElement",
    ];

    // Ops that accept sequences and empty optionals as inputs besides the sequence ops.
    private static readonly HashSet<string> s_sequenceInputOps = ["Identity", "If", "Loop", "Scan"];

    /// <summary>Translates a sequence or optional op, or an Identity of a sequence or optional. Returns false for other nodes.</summary>
    private bool TrySequenceOp(OnnxNode n, TranslatedTensor?[] ins)
    {
        bool nonTensor = ins.Any(t => t is { IsTensor: false });
        if (!s_sequenceOps.Contains(n.OpType) && !nonTensor)
        {
            return false;
        }

        if (!s_sequenceOps.Contains(n.OpType) && !s_sequenceInputOps.Contains(n.OpType))
        {
            throw new NotSupportedException($"{n.OpType} of a sequence or optional value");
        }

        if (n.OpType is "If" or "Loop" or "Scan")
        {
            return false;   // control flow passes sequences and optionals through its subgraphs itself
        }

        if (n.OpType == "SequenceMap")
        {
            SequenceMap(n, ins);
            return true;
        }

        _env[n.Outputs[0]] = n.OpType switch
        {
            "OptionalHasElement" => BoolConst(Opt(ins, 0) is { IsNone: false }),
            "SequenceEmpty" => TranslatedTensor.Seq([]),
            "SequenceConstruct" => TranslatedTensor.Seq([.. ins.Select((t, i) => Element(t, i))]),
            "SequenceInsert" => SequenceInsert(ins),
            "SequenceErase" => SequenceErase(ins),
            "SequenceAt" => SequenceAt(ins),
            "SequenceLength" => SequenceLength(ins[0]!),
            "SplitToSequence" => SplitToSequence(n, ins),
            "ConcatFromSequence" => ConcatFromSequence(n, ins[0]!),
            _ => OptionalOp(n, ins),   // Identity, Optional, OptionalGetElement
        };
        return true;
    }

    private TranslatedTensor OptionalOp(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = Opt(ins, 0);
        return n.OpType switch
        {
            "Optional" => x ?? TranslatedTensor.None(),
            "OptionalGetElement" when x is null or { IsNone: true } => throw new NotSupportedException("OptionalGetElement of an empty optional"),
            _ => x!,
        };
    }

    private static TranslatedTensor BoolConst(bool value) => TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Bool, [], [value ? 1 : 0]));

    private static TranslatedTensor Element(TranslatedTensor? t, int i)
        => t is { IsTensor: true } ? t : throw new NotSupportedException($"sequence element {i} is not a tensor (nested sequences and optionals are not supported)");

    /// <summary>The elements of a sequence known at import time.</summary>
    private static IReadOnlyList<TranslatedTensor> Elements(TranslatedTensor t, string what)
        => t.Sequence ?? throw new NotSupportedException(t.Stacked ? $"{what} of a sequence whose length is only known at run time" : $"{what} of a value that is not a sequence");

    /// <summary>A constant position into a sequence of <paramref name="count"/> elements (negative counts from the end).</summary>
    private int Position(TranslatedTensor?[] ins, int i, int count, int defaultPosition, bool insert)
    {
        if (Opt(ins, i) is null)
        {
            return defaultPosition;
        }

        long p = ConstIn(ins, i, "sequence position").ScalarInt();
        p = p < 0 ? p + count : p;
        return p < 0 || p > count || (!insert && p == count) ? throw new InvalidDataException($"sequence position {p} out of range for {count} elements") : (int)p;
    }

    private TranslatedTensor SequenceInsert(TranslatedTensor?[] ins)
    {
        var tensor = Element(ins[1], 0);
        if (ins[0]!.Stacked)
        {
            if (Opt(ins, 2) is not null)
            {
                throw new NotSupportedException("SequenceInsert at a position into a sequence whose length is only known at run time");
            }

            var row = Synthetic("Unsqueeze", [tensor], IntsAttribute("axes", 0));
            return new TranslatedTensor { V = Synthetic("Concat", [TranslatedTensor.Dyn(ins[0]!.V!), row], IntAttribute("axis", 0)).V, Stacked = true };
        }

        var list = Elements(ins[0]!, "SequenceInsert").ToList();
        list.Insert(Position(ins, 2, list.Count, list.Count, insert: true), tensor);
        return TranslatedTensor.Seq(list);
    }

    private TranslatedTensor SequenceErase(TranslatedTensor?[] ins)
    {
        var list = Elements(ins[0]!, "SequenceErase").ToList();
        list.RemoveAt(Position(ins, 1, list.Count, list.Count - 1, insert: false));
        return TranslatedTensor.Seq(list);
    }

    private TranslatedTensor SequenceAt(TranslatedTensor?[] ins)
    {
        var seq = ins[0]!;
        if (seq.Sequence is { } list && ins[1]!.IsConst)
        {
            return list[Position(ins, 1, list.Count, 0, insert: false)];
        }

        // A runtime position (or a runtime-length sequence): gather one row of the stacked elements.
        var stacked = seq.Stacked ? seq.V! : AsStacked(Elements(seq, "SequenceAt"));
        if (ins[1]!.IsConst && ins[1]!.C!.ScalarInt() < 0 && seq.Stacked)
        {
            throw new NotSupportedException("SequenceAt with a negative position into a sequence whose length is only known at run time");
        }

        var row = _b.Emit1("GatherAxis", [stacked, Idx(ins[1]!)], Attributes.From(("axis", 0L)));
        var element = new Shape([.. stacked.Shape.Dims.Skip(1)]);
        bool scalar = element.Rank == 0;
        return new TranslatedTensor { V = Rs(row, scalar ? [1L] : [.. element.Dims.Select(d => d.Fixed)]), Scalar = scalar };
    }

    /// <summary>Stacks same-shaped elements along a new outer axis.</summary>
    private Value AsStacked(IReadOnlyList<TranslatedTensor> elements)
    {
        if (elements.Count == 0)
        {
            throw new NotSupportedException("indexing an empty sequence");
        }

        return AsValueKeepingType(Stack([.. elements], 0));
    }

    private Value AsValueKeepingType(TranslatedTensor t) => t.V ?? (OnnxType.IsFloat(t.C!.ElemType) ? AsValue(t) : Idx(t));

    private static TranslatedTensor SequenceLength(TranslatedTensor seq)
    {
        long length = seq.Stacked ? SymbolicInts.Encode(seq.V!.Shape[0]) : Elements(seq, "SequenceLength").Count;
        return TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [], [length]));
    }

    private TranslatedTensor SplitToSequence(OnnxNode n, TranslatedTensor?[] ins)
    {
        var x = Element(ins[0], 0);
        int rank = OnnxRank(x);
        int axis = NormalizeAxis(n.Int("axis", 0), rank);
        long extent = StaticExtent(x, axis) ?? throw new NotSupportedException("SplitToSequence along a symbolic axis");
        var split = OptConst(ins, 1, "SplitToSequence split");
        bool squeeze = split is null && n.Int("keepdims", 1) == 0;
        List<long> sizes;
        if (split is null || split.Rank == 0)
        {
            long size = split?.ScalarInt() ?? 1;
            sizes = [.. Enumerable.Range(0, (int)((extent + size - 1) / size)).Select(k => Math.Min(size, extent - k * size))];
        }
        else
        {
            sizes = [.. split.Ints()];
        }

        var parts = new List<TranslatedTensor>();
        long start = 0;
        foreach (var size in sizes)
        {
            var part = Synthetic("Slice", [x, IntsConst(start), IntsConst(start + size), IntsConst(axis)]);
            parts.Add(squeeze ? Synthetic("Squeeze", [part, IntsConst(axis)]) : part);
            start += size;
        }

        return TranslatedTensor.Seq(parts);
    }

    private static TranslatedTensor IntsConst(params long[] values) => TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int64, [values.Length], values));

    private TranslatedTensor ConcatFromSequence(OnnxNode n, TranslatedTensor seq)
    {
        bool newAxis = n.Int("new_axis", 0) == 1;
        if (seq.Stacked)
        {
            return ConcatStacked(seq.V!, n.Int("axis", 0), newAxis);
        }

        var elements = Elements(seq, "ConcatFromSequence");
        if (elements.Count == 0)
        {
            throw new NotSupportedException("ConcatFromSequence of an empty sequence");
        }

        int rank = OnnxRank(elements[0]) + (newAxis ? 1 : 0);
        long axis = NormalizeAxis(n.Int("axis", 0), rank);
        var parts = newAxis ? elements.Select(e => Synthetic("Unsqueeze", [e], IntsAttribute("axes", axis))).ToArray() : [.. elements];
        return parts.Length == 1 ? parts[0] : Synthetic("Concat", parts, IntAttribute("axis", axis));
    }

    /// <summary>Concatenation of a runtime-length sequence: possible along the outer axis only, where the elements already lie.</summary>
    private TranslatedTensor ConcatStacked(Value stacked, long axis, bool newAxis)
    {
        if (axis != 0)
        {
            throw new NotSupportedException("ConcatFromSequence of a sequence whose length is only known at run time along an inner axis");
        }

        if (newAxis)
        {
            return TranslatedTensor.Dyn(stacked);
        }

        if (stacked.Rank < 2)
        {
            throw new NotSupportedException("ConcatFromSequence of scalars");
        }

        var outer = stacked.Shape[0] * stacked.Shape[1];
        long[] shape = [0, .. stacked.Shape.Dims.Skip(2).Select(d => d.Fixed)];
        return TranslatedTensor.Dyn(Emit("Reshape", [stacked], Attributes.From(("shape", shape), ("outer_symbol", outer.ToString()))));
    }

    /// <summary>SequenceMap: the body runs once per element at import time (like an unrolled loop); tensor inputs are passed to every run.</summary>
    private void SequenceMap(OnnxNode n, TranslatedTensor?[] ins)
    {
        var body = n.Attributes["body"].G!;
        int count = Elements(ins[0]!, "SequenceMap").Count;
        var outputs = body.Outputs.Select(_ => new List<TranslatedTensor>()).ToArray();
        for (int i = 0; i < count; i++)
        {
            for (int j = 0; j < ins.Length; j++)
            {
                var input = ins[j]!;
                _env[body.Inputs[j].Name] = input.IsTensor ? input : Elements(input, "SequenceMap")[i];
            }

            RunSubgraph(body, "SequenceMap body");
            for (int k = 0; k < outputs.Length; k++)
            {
                outputs[k].Add(Get(body.Outputs[k].Name));
            }
        }

        for (int k = 0; k < outputs.Length; k++)
        {
            if (n.Outputs[k] != "")
            {
                _env[n.Outputs[k]] = TranslatedTensor.Seq(outputs[k]);
            }
        }
    }
}
