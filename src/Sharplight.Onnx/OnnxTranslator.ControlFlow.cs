namespace Sharplight.Onnx;

// Control flow. If with a compile-time constant condition is inlined. Loop and Scan run their body at import time when they can: bodies
// that only compute constants are evaluated (e.g. TF "arange" loops), bodies that depend on runtime tensors are unrolled up to
// OnnxImportOptions.MaxUnrolledLoop iterations. Everything else (runtime trip counts or conditions, longer runtime loops, symbolic scan
// lengths) becomes the runtime IR Loop (OnnxTranslator.RuntimeLoop.cs).
internal sealed partial class OnnxTranslator
{
    private int _syntheticValues;

    // ---- If --------------------------------------------------------------------------------------------------------
    private void IfOp(OnnxNode n, TranslatedTensor?[] ins)
    {
        if (!ins[0]!.IsConst)
        {
            RuntimeIf(n, ins);
            return;
        }

        var cond = ConstIn(ins, 0, "If condition");
        var g = (cond.Floats()[0] != 0 ? n.Attributes["then_branch"] : n.Attributes["else_branch"]).G!;
        RunSubgraph(g, "If branch");
        for (int i = 0; i < n.Outputs.Count; i++)
        {
            if (n.Outputs[i] != "")
            {
                _env[n.Outputs[i]] = Get(g.Outputs[i].Name);
            }
        }
    }

    // ---- Loop ------------------------------------------------------------------------------------------------------
    private void LoopOp(OnnxNode n, TranslatedTensor?[] ins)
    {
        var body = n.Attributes["body"].G!;
        var trip = Opt(ins, 0);
        var condition = Opt(ins, 1);
        var carried = ins.Skip(2).Select(t => t!).ToArray();
        var runtime = RuntimeValues(body, carried.Select((t, k) => (body.Inputs[2 + k].Name, body.Outputs[1 + k].Name, IsRuntime(t))).ToArray());
        bool runtimeBody = body.Outputs.Skip(1).Any(o => runtime.Contains(o.Name));
        bool knownTrip = trip is null || trip is { IsConst: true, C.HasSymbols: false };
        long tripCount = trip is null ? long.MaxValue : knownTrip ? trip.C!.ScalarInt() : 0;
        if (!knownTrip || condition is { IsConst: false } || runtime.Contains(body.Outputs[0].Name) || (runtimeBody && tripCount > _opts.MaxUnrolledLoop))
        {
            RuntimeLoop(n, body, trip, condition, carried, runtime);
            return;
        }

        bool running = condition is null || condition.C!.Ints()[0] != 0;
        int scanCount = body.Outputs.Count - 1 - carried.Length;
        var scans = Enumerable.Range(0, scanCount).Select(_ => new List<TranslatedTensor>()).ToArray();
        for (long i = 0; i < tripCount && running; i++)
        {
            CheckIterations(i, runtimeBody, "Loop");
            _env[body.Inputs[0].Name] = TranslatedTensor.Const(OnnxConst.ScalarInt(i));
            _env[body.Inputs[1].Name] = TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Bool, [], [1]));
            for (int k = 0; k < carried.Length; k++)
            {
                _env[body.Inputs[2 + k].Name] = carried[k];
            }

            RunSubgraph(body, "Loop body");
            var cond = Get(body.Outputs[0].Name);
            running = cond.IsConst
                ? cond.C!.Ints()[0] != 0
                : throw new NotSupportedException("Loop whose condition depends on runtime tensors");
            carried = Enumerable.Range(0, carried.Length).Select(k => Get(body.Outputs[1 + k].Name)).ToArray();
            for (int j = 0; j < scanCount; j++)
            {
                scans[j].Add(Get(body.Outputs[1 + carried.Length + j].Name));
            }
        }

        SetOutputs(n, carried, scans.Select(s => Stack(s, 0)).ToArray());
    }

    // ---- Scan ------------------------------------------------------------------------------------------------------
    private void ScanOp(OnnxNode n, TranslatedTensor?[] ins)
    {
        var body = n.Attributes["body"].G!;
        if (Opset < 9)
        {
            Scan8(n, body, ins);
            return;
        }

        int scanInputs = (int)n.Int("num_scan_inputs", 0);
        int stateCount = ins.Length - scanInputs;
        int scanOutputs = body.Outputs.Count - stateCount;
        var scan = new ScanSpec(
            n.Ints("scan_input_axes") ?? new long[scanInputs],
            n.Ints("scan_input_directions") ?? new long[scanInputs],
            n.Ints("scan_output_axes") ?? new long[scanOutputs],
            n.Ints("scan_output_directions") ?? new long[scanOutputs]);
        var (states, stacked) = Scan(n, body, [.. ins.Take(stateCount).Select(t => t!)], [.. ins.Skip(stateCount).Select(t => t!)], scan);
        SetOutputs(n, states, stacked);
    }

    /// <summary>Scan-8: a batch axis 0 in front of the scan axis and an optional <c>sequence_lens</c> input; each batch entry is scanned on its own.</summary>
    private void Scan8(OnnxNode n, OnnxGraph body, TranslatedTensor?[] ins)
    {
        if (ins[0] is not null)
        {
            throw new NotSupportedException("Scan-8 with sequence_lens");
        }

        int scanInputs = (int)n.Int("num_scan_inputs", 0);
        int stateCount = ins.Length - 1 - scanInputs;
        int scanOutputs = body.Outputs.Count - stateCount;
        var scan = new ScanSpec(new long[scanInputs], n.Ints("directions") ?? new long[scanInputs], new long[scanOutputs], new long[scanOutputs]);
        var batch = Extent(ins[1]!, 0);
        var states = new List<TranslatedTensor>[stateCount];
        var outputs = new List<TranslatedTensor>[scanOutputs];
        for (long b = 0; b < batch; b++)
        {
            var entry = ins.Skip(1).Select(t => Synthetic("Gather", [t!, TranslatedTensor.Const(OnnxConst.ScalarInt(b))], IntAttribute("axis", 0))).ToArray();
            var (s, o) = Scan(n, body, entry[..stateCount], entry[stateCount..], scan);
            for (int k = 0; k < stateCount; k++)
            {
                (states[k] ??= []).Add(s[k]);
            }

            for (int j = 0; j < scanOutputs; j++)
            {
                (outputs[j] ??= []).Add(o[j]);
            }
        }

        SetOutputs(n, [.. states.Select(s => Stack(s, 0))], [.. outputs.Select(s => Stack(s, 0))]);
    }

    /// <summary>One Scan over the given states and scan inputs: unrolled or evaluated at import time when possible, else a runtime loop.</summary>
    private (TranslatedTensor[] States, TranslatedTensor[] Outputs) Scan(OnnxNode n, OnnxGraph body, TranslatedTensor[] states, TranslatedTensor[] sequences, ScanSpec scan)
    {
        int stateCount = states.Length;
        int scanOutputs = body.Outputs.Count - stateCount;
        var axes = sequences.Select((s, j) => NormalizeAxis(scan.InputAxes[j], OnnxRank(s))).ToArray();
        var runtime = RuntimeValues(
            body,
            [
                .. states.Select((t, k) => (body.Inputs[k].Name, body.Outputs[k].Name, IsRuntime(t))),
                .. sequences.Select((t, j) => (body.Inputs[stateCount + j].Name, "", IsRuntime(t))),
            ]);
        bool runtimeBody = body.Outputs.Any(o => runtime.Contains(o.Name));
        long? length = sequences.Length == 0 ? 0 : StaticExtent(sequences[0], axes[0]);
        if (length is null || (runtimeBody && length > _opts.MaxUnrolledLoop))
        {
            return RuntimeScan(n, body, states, sequences, axes, scan);
        }

        var scans = Enumerable.Range(0, scanOutputs).Select(_ => new List<TranslatedTensor>()).ToArray();
        for (long t = 0; t < length; t++)
        {
            CheckIterations(t, runtimeBody, "Scan");
            for (int k = 0; k < stateCount; k++)
            {
                _env[body.Inputs[k].Name] = states[k];
            }

            for (int j = 0; j < sequences.Length; j++)
            {
                long index = scan.InputDirections[j] == 1 ? length.Value - 1 - t : t;
                _env[body.Inputs[stateCount + j].Name] = Synthetic("Gather", [sequences[j], TranslatedTensor.Const(OnnxConst.ScalarInt(index))], IntAttribute("axis", axes[j]));
            }

            RunSubgraph(body, "Scan body");
            states = Enumerable.Range(0, stateCount).Select(k => Get(body.Outputs[k].Name)).ToArray();
            for (int j = 0; j < scanOutputs; j++)
            {
                scans[j].Add(Get(body.Outputs[stateCount + j].Name));
            }
        }

        var stacked = scans.Select((s, j) =>
        {
            var items = scan.OutputDirections[j] == 1 ? Enumerable.Reverse(s).ToList() : s;
            return Stack(items, scan.OutputAxes[j] < 0 ? scan.OutputAxes[j] + OnnxRank(s[0]) + 1 : scan.OutputAxes[j]);
        }).ToArray();
        return (states, stacked);
    }

    private void CheckIterations(long done, bool runtimeBody, string op)
    {
        if (done >= _opts.MaxLoopIterations)
        {
            throw new NotSupportedException($"{op} runs more than {_opts.MaxLoopIterations} iterations at import time (OnnxImportOptions.MaxLoopIterations)");
        }

        if (runtimeBody && done >= _opts.MaxUnrolledLoop)
        {
            throw new NotSupportedException($"{op} with a body that depends on runtime tensors has more than {_opts.MaxUnrolledLoop} iterations (OnnxImportOptions.MaxUnrolledLoop)");
        }
    }

    private void SetOutputs(OnnxNode n, TranslatedTensor[] states, TranslatedTensor[] scans)
    {
        for (int i = 0; i < n.Outputs.Count; i++)
        {
            if (n.Outputs[i] != "")
            {
                _env[n.Outputs[i]] = i < states.Length ? states[i] : scans[i - states.Length];
            }
        }
    }

    /// <summary>Translates the nodes of a subgraph into the current environment (outer values are visible by name).</summary>
    private void RunSubgraph(OnnxGraph g, string what)
    {
        foreach (var t in g.Initializers)
        {
            if (t.Source is not null)
            {
                _env[t.Name] = TranslatedTensor.Const(OnnxConst.FromTensor(t));
            }
        }

        foreach (var sub in g.Nodes)
        {
            if (!IsSupportedNode(sub))
            {
                throw new NotSupportedException($"{what} uses unsupported operator {sub.OpType}");
            }

            HandleNode(sub);
        }
    }

    /// <summary>Stacks per-iteration values along a new axis (folded when they are all constant).</summary>
    private TranslatedTensor Stack(List<TranslatedTensor> items, long axis)
    {
        if (items.Count == 0)
        {
            throw new NotSupportedException("a Loop or Scan with zero iterations produces an empty scan output");
        }

        var parts = items.Select(t => Synthetic("Unsqueeze", [t], IntsAttribute("axes", axis))).ToArray();
        return parts.Length == 1 ? parts[0] : Synthetic("Concat", parts, IntAttribute("axis", axis));
    }

    /// <summary>Translates one synthetic ONNX node over already translated values (folding it when its inputs are constant).</summary>
    private TranslatedTensor Synthetic(string op, TranslatedTensor[] inputs, params OnnxAttribute[] attributes)
    {
        var names = inputs.Select(t =>
        {
            string name = $"#synthetic{_syntheticValues++}";
            _env[name] = t;
            return name;
        }).ToArray();
        string output = $"#synthetic{_syntheticValues++}";
        HandleNode(new OnnxNode { Name = output, OpType = op, Inputs = names, Outputs = [output], Attributes = attributes.ToDictionary(a => a.Name) });
        return Get(output);
    }

    private int OnnxRank(TranslatedTensor t) => t.IsConst ? t.C!.Rank : t.V!.Rank + t.LeadingUnits;

    private static long Extent(TranslatedTensor t, int axis)
        => StaticExtent(t, axis) ?? throw new NotSupportedException("Scan-8 over a symbolic batch axis");

    /// <summary>The ONNX extent of <paramref name="axis"/>, or null when it is symbolic.</summary>
    private static long? StaticExtent(TranslatedTensor t, int axis)
    {
        if (t.IsConst)
        {
            return t.C!.Dims[axis];
        }

        int irAxis = axis - t.LeadingUnits;
        if (irAxis < 0)
        {
            return 1;
        }

        var d = t.V!.Shape[irAxis];
        return d.IsSymbolic ? null : d.Fixed;
    }

    private static OnnxAttribute IntAttribute(string name, long value) => new() { Name = name, Type = OnnxAttrType.Int, I = value };

    private static OnnxAttribute IntsAttribute(string name, params long[] values) => new() { Name = name, Type = OnnxAttrType.Ints, Ints = values };
}
