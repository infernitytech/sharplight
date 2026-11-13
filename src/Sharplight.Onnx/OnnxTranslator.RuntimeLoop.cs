using Sharplight.Graph;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Onnx;

// Loop and Scan that cannot run at import time: the body becomes a ComputeGraph (built with its own GraphBuilder like an If branch) and
// the IR Loop node (StandardOps.Loop) runs it once per iteration. Loop-carried values keep their type and shape across iterations; scan
// outputs are stacked along a new outermost axis whose extent is the trip count when it is known, else a dynamic symbol.
internal sealed partial class OnnxTranslator
{
    private int _loopCount;

    /// <summary>Whether a value is only known at run time.</summary>
    private static bool IsRuntime(TranslatedTensor t) => !t.IsConst;

    /// <summary>
    /// The names in <paramref name="body"/> whose values depend on runtime tensors: enclosing values that are runtime, body inputs whose
    /// initial value is runtime (<paramref name="inputs"/>: body input, the body output that carries it into the next iteration or empty,
    /// runtime initial value), and everything computed from them. Carried values become runtime once their update is (fixed point).
    /// Shapes and sizes count as known: they are constant per iteration.
    /// </summary>
    private HashSet<string> RuntimeValues(OnnxGraph body, (string Input, string Output, bool Runtime)[] inputs)
    {
        var runtimeInputs = inputs.Where(i => i.Runtime).Select(i => i.Input).ToHashSet(StringComparer.Ordinal);
        var outer = FreeVariables(body).Where(name => _env.TryGetValue(name, out var t) && IsRuntime(t)).ToList();
        while (true)
        {
            var runtime = new HashSet<string>(runtimeInputs.Concat(outer), StringComparer.Ordinal);
            foreach (var node in body.Nodes)
            {
                if (node.OpType is "Shape" or "Size" or "SequenceLength" or "OptionalHasElement")
                {
                    continue;
                }

                var captured = node.Attributes.Values.Where(a => a.G is not null).SelectMany(a => FreeVariables(a.G!));
                if (node.Inputs.Concat(captured).Any(runtime.Contains))
                {
                    runtime.UnionWith(node.Outputs.Where(o => o != ""));
                }
            }

            int before = runtimeInputs.Count;
            runtimeInputs.UnionWith(inputs.Where(i => i.Output != "" && runtime.Contains(i.Output)).Select(i => i.Input));
            if (runtimeInputs.Count == before)
            {
                return runtime;
            }
        }
    }

    // ---- Loop ------------------------------------------------------------------------------------------------------
    private void RuntimeLoop(OnnxNode n, OnnxGraph body, TranslatedTensor? trip, TranslatedTensor? condition, TranslatedTensor[] carried, HashSet<string> runtime)
    {
        Value tripValue;
        Dim? knownTrip = null;
        string tripSymbol = "";
        if (trip is null)
        {
            tripValue = LoopConstant(int.MaxValue);
        }
        else if (trip.IsConst)
        {
            long code = trip.C!.Ints()[0];
            knownTrip = SymbolicInts.ToDim(code);
            tripSymbol = knownTrip.Value.IsSymbolic ? knownTrip.Value.Symbol! : "";
            tripValue = LoopConstant(knownTrip.Value.IsSymbolic ? int.MaxValue : Math.Clamp(code, 0, int.MaxValue));
        }
        else
        {
            tripValue = Idx(trip);
        }

        bool startsTrue = condition is null || (condition.IsConst && condition.C!.Floats()[0] != 0);
        var condValue = condition is { IsConst: false } ? condition.V! : LoopConstant(startsTrue ? 1 : 0);
        int scanCount = body.Outputs.Count - 1 - carried.Length;
        bool fixedCount = knownTrip is not null && startsTrue && KeepsCondition(body) && !runtime.Contains(body.Outputs[0].Name);
        string prefix = $"loop{_loopCount++}.";
        string countSymbol = fixedCount || scanCount == 0 ? "" : NewDynamicSymbol(prefix + "count", knownTrip ?? _opts.RuntimeLoopScanBound);
        var built = BuildLoopBody(
            body, prefix, body.Inputs[0].Name, body.Inputs[1].Name, [.. carried.Select((t, k) => (body.Inputs[2 + k].Name, t))], [], body.Outputs[0].Name,
            [.. body.Outputs.Skip(1).Take(carried.Length).Select(o => o.Name)], [.. body.Outputs.Skip(1 + carried.Length).Select(o => o.Name)]);
        var spec = new ScanSpec([], [], new long[scanCount], new long[scanCount]);
        var extent = countSymbol.Length > 0 ? countSymbol : knownTrip?.ToString() ?? "0";
        var (states, outputs) = EmitLoop(n, built, tripValue, condValue, carried, [], spec, tripSymbol, extent, countSymbol);
        SetOutputs(n, states, outputs);
    }

    /// <summary>Whether the body passes its condition input through unchanged, so the loop runs exactly the trip count.</summary>
    private static bool KeepsCondition(OnnxGraph body)
    {
        string input = body.Inputs[1].Name;
        string output = body.Outputs[0].Name;
        return output == input || body.Nodes.Any(node => node.OpType == "Identity" && node.Inputs[0] == input && node.Outputs[0] == output);
    }

    // ---- Scan ------------------------------------------------------------------------------------------------------
    private (TranslatedTensor[] States, TranslatedTensor[] Outputs) RuntimeScan(
        OnnxNode n, OnnxGraph body, TranslatedTensor[] states, TranslatedTensor[] sequences, int[] axes, ScanSpec scan)
    {
        var scanInputs = new TranslatedTensor[sequences.Length];
        for (int j = 0; j < sequences.Length; j++)
        {
            var v = LoopValue(sequences[j], "a scan input");
            var perm = Enumerable.Range(0, v.Rank).Where(d => d != axes[j]).Prepend(axes[j]).ToArray();
            scanInputs[j] = TranslatedTensor.Dyn(Permute(v, perm));
        }

        var extent = scanInputs[0].V!.Shape[0];
        if (scanInputs.Any(s => s.V!.Shape[0] != extent))
        {
            throw new NotSupportedException("Scan inputs of different lengths");
        }

        int stateCount = states.Length;
        string prefix = $"scan{_loopCount++}.";
        var slices = scanInputs.Select((s, j) => (body.Inputs[stateCount + j].Name, s.V!.Type, s.V.Shape)).ToArray();
        var built = BuildLoopBody(
            body, prefix, null, null, [.. states.Select((t, k) => (body.Inputs[k].Name, t))], slices, null,
            [.. body.Outputs.Take(stateCount).Select(o => o.Name)], [.. body.Outputs.Skip(stateCount).Select(o => o.Name)]);
        var outputs = EmitLoop(n, built, LoopConstant(int.MaxValue), LoopConstant(1), states, scanInputs, scan, "", extent.ToString(), "");
        return outputs;
    }

    // ---- shared ----------------------------------------------------------------------------------------------------

    /// <summary>
    /// Translates a loop body into its own graph. Inputs: the iteration number and condition (when named), the loop-carried values (typed
    /// and shaped like their initial values), one slice of each scan input (the input shape without its outermost axis) and the captured
    /// enclosing values. Outputs: the condition (when named), the carried values (checked against their inputs) and the scan outputs.
    /// </summary>
    private LoopBody BuildLoopBody(
        OnnxGraph body, string prefix, string? iteration, string? condition, (string Name, TranslatedTensor Initial)[] states,
        (string Name, DataType Type, Shape Shape)[] slices, string? conditionOutput, string[] stateOutputs, string[] scanOutputs)
    {
        var free = FreeVariables(body).Where(_env.ContainsKey).Distinct().OrderBy(s => s, StringComparer.Ordinal).ToList();
        var captured = free.Where(name => !_env[name].IsConst).ToList();
        foreach (var name in captured)
        {
            LoopValue(_env[name], $"captured value '{name}'");
        }

        var initial = states.Select(s => LoopValue(s.Initial, "a loop-carried value")).ToArray();
        var scalarScans = new bool[scanOutputs.Length];
        var graph = BuildSubgraph(free, prefix, () =>
        {
            if (iteration is not null)
            {
                _env[iteration] = new TranslatedTensor { V = _b.Input(iteration, InputRole.Other, DataType.I32, new Shape([(Dim)1L])), Scalar = true };
            }

            if (condition is not null)
            {
                _env[condition] = new TranslatedTensor { V = _b.Input(condition, InputRole.Other, DataType.F32, new Shape([(Dim)1L])), Scalar = true };
            }

            for (int k = 0; k < states.Length; k++)
            {
                var v = _b.Input(states[k].Name, InputRole.Other, initial[k].Type, initial[k].Shape);
                _env[states[k].Name] = new TranslatedTensor { V = v, Scalar = IsScalar(states[k].Initial) };
            }

            foreach (var (name, type, shape) in slices)
            {
                bool scalar = shape.Rank == 1;
                var sliceShape = scalar ? new Shape([(Dim)1L]) : new Shape([.. shape.Dims.Skip(1)]);
                _env[name] = new TranslatedTensor { V = _b.Input(name, InputRole.Other, type, sliceShape), Scalar = scalar };
            }

            RunSubgraph(body, "Loop body");
            if (conditionOutput is not null)
            {
                _b.Output("cond_out", OutputCopy(LoopValue(Get(conditionOutput), "the loop condition"), prefix + "cond_out"));
            }

            for (int k = 0; k < stateOutputs.Length; k++)
            {
                var v = LoopValue(Get(stateOutputs[k]), "a loop-carried value");
                if (v.Type != initial[k].Type)
                {
                    v = _b.Emit1("Cast", [v], Attributes.From(("type", initial[k].Type.Name)));
                }

                if (!v.Shape.Dims.SequenceEqual(initial[k].Shape.Dims))
                {
                    throw new NotSupportedException($"loop-carried value '{states[k].Name}' changes its shape from {initial[k].Shape} to {v.Shape} between iterations");
                }

                _b.Output($"state{k}", OutputCopy(v, $"{prefix}state{k}"));
            }

            for (int j = 0; j < scanOutputs.Length; j++)
            {
                var t = Get(scanOutputs[j]);
                var v = LoopValue(t, "a scan output");
                if (v.Shape.Dims.Any(d => d.IsSymbolic))
                {
                    throw new NotSupportedException($"scan output '{scanOutputs[j]}' has a symbolic dimension {v.Shape}; it would not be the outermost after stacking");
                }

                scalarScans[j] = IsScalar(t);
                _b.Output($"scan{j}", OutputCopy(v, $"{prefix}scan{j}"));
            }
        });

        string[] inputs = [iteration ?? "", condition ?? "", .. states.Select(s => s.Name), .. slices.Select(s => s.Name), .. captured];
        string[] outputs = [conditionOutput is null ? "" : "cond_out", .. stateOutputs.Select((_, k) => $"state{k}"), .. scanOutputs.Select((_, j) => $"scan{j}")];
        return new LoopBody(graph, inputs, outputs, captured, scalarScans);
    }

    /// <summary>Emits the IR Loop node and binds the ONNX outputs (carried values, then scan outputs stacked along <paramref name="scan"/> axes).</summary>
    private (TranslatedTensor[] States, TranslatedTensor[] Outputs) EmitLoop(
        OnnxNode n, LoopBody body, Value trip, Value condition, TranslatedTensor[] states, TranslatedTensor[] scanInputs, ScanSpec scan,
        string tripSymbol, string scanExtent, string countSymbol)
    {
        Value[] inputs =
        [
            trip, condition, .. states.Select(t => HostReadable(LoopValue(t, "a loop-carried value"))), .. scanInputs.Select(t => t.V!),
            .. body.Captured.Select(name => _env[name].V!),
        ];
        var attributes = Attributes.From(
            ("body", body.Graph), ("body_inputs", body.Inputs), ("body_outputs", body.Outputs), ("state_count", (long)states.Length),
            ("scan_input_count", (long)scanInputs.Length), ("scan_input_reverse", scan.InputDirections), ("scan_output_reverse", scan.OutputDirections),
            ("trip_symbol", tripSymbol), ("scan_extent", scanExtent), ("count_symbol", countSymbol));
        var outputs = _b.Emit("Loop", inputs, attributes, Nm(n));
        var carried = states.Select((t, k) => new TranslatedTensor { V = outputs[k], Scalar = IsScalar(t) }).ToArray();
        var stacked = new TranslatedTensor[outputs.Count - states.Length];
        for (int j = 0; j < stacked.Length; j++)
        {
            var v = outputs[states.Length + j];
            if (body.ScalarScans[j])
            {
                v = Emit("Reshape", [v], Attributes.From(("shape", L(Fixed(v)[0]))));
            }

            int axis = scan.OutputAxes[j] < 0 ? (int)scan.OutputAxes[j] + v.Rank : (int)scan.OutputAxes[j];
            if (axis != 0)
            {
                var perm = Enumerable.Range(1, v.Rank - 1).ToList();
                perm.Insert(axis, 0);
                v = Permute(v, [.. perm]);
            }

            stacked[j] = TranslatedTensor.Dyn(v);
        }

        return (carried, stacked);
    }

    /// <summary>The IR value a loop reads or produces for a tensor (constants are materialized with their integer or float type).</summary>
    private Value LoopValue(TranslatedTensor t, string what)
    {
        if (t.LeadingUnits > 0 || t.TransposedNonZero)
        {
            throw new NotSupportedException($"{what} of a runtime loop is a tensor with leading unit dims or a NonZero result");
        }

        return t.V ?? (OnnxType.IsFloat(t.C!.ElemType) ? AsValue(t) : Idx(t));
    }

    /// <summary>Host segments read graph constants directly; a large constant stored as a weight is copied into an intermediate first.</summary>
    private Value HostReadable(Value v) => v.Kind == ValueKind.Weight ? OutputCopy(v, $"{v.Name}.copy") : v;

    private Value LoopConstant(long value) => Idx(TranslatedTensor.Const(OnnxConst.FromInts(OnnxType.Int32, [], [value])));

    private static bool IsScalar(TranslatedTensor t) => t.Scalar || (t.IsConst && t.C!.Rank == 0);
}
