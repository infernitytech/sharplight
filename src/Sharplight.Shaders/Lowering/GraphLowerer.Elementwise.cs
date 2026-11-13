using System.Text;
using Sharplight.Runtime;
using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    // ---- elementwise codegen -----------------------------------------------------------------
    private void EmitCopy(TensorRef src, TensorRef dst, string name, IEnumerable<Node>? nodes = null)
    {
        if (src.Type == dst.Type && src.Type != DataType.F32)
        {
            EmitRawCopy(src, dst, name, nodes ?? []);
            return;
        }

        if (dst.Type != DataType.F32)
        {
            throw new NotSupportedException($"copy {src.Type} → {dst.Type}");
        }

        EmitElementwiseKernel(name, nodes ?? [], dst.Shape, dst.Symbol, [src], [(dst, "a0")], []);
    }

    /// <summary>
    /// Lowers a fused elementwise group (optionally with a row norm of one of its outputs after it, or of one of its
    /// inputs before it) into one generated kernel.
    /// </summary>
    private void LowerElementwise(List<Node> group, List<Value> outputs, Node? normTail = null, Node? normHead = null)
    {
        var shapeValue = group[0].Output.Shape;
        var inputs = new List<TensorRef>();
        var inputIndex = new Dictionary<Value, int>();
        var statements = new List<string>();
        var names = new Dictionary<Value, string>();

        string Operand(Value v)
        {
            if (names.TryGetValue(v, out var existing))
            {
                return existing;
            }

            if (!inputIndex.TryGetValue(v, out int index))
            {
                index = inputs.Count;
                inputs.Add(Ref(v));
                inputIndex[v] = index;
            }

            return names[v] = "a" + index;
        }

        string? headInput = null;
        if (normHead is not null)
        {
            headInput = Operand(normHead.Inputs[0]);
            names[normHead.Output] = "nrm";
        }

        foreach (var node in group)
        {
            var args = node.Inputs.Select(Operand).ToArray();
            string expression = OpExpr(node, args);
            var local = "v" + statements.Count;
            statements.Add($"float {local} = {expression};");
            names[node.Output] = local;
        }

        var outs = outputs.Select(o => (Produce(o), names[o])).ToList();
        var shape = BoundShape(shapeValue);
        var label = string.Join(",", group.Select(g => g.Op.Name));
        if (normTail is not null)
        {
            TensorRef? weight = normTail.Inputs.Length > 1 ? Ref(normTail.Inputs[1]) : null;
            EmitRowNormKernel($"fused[{label}]+{normTail.Op.Name}", [.. group, normTail], shape, shapeValue.Symbol,
                inputs, preStmts: statements, normExpr: names[normTail.Inputs[0]], pass1Outputs: outs, postStmts: [],
                pass2Outputs: [(Produce(normTail.Output), "nrm")], weight, normTail.Op.Name == "RmsNorm" ? 0 : 1, normTail.Attributes.Get<float>("eps"));
            return;
        }

        if (normHead is not null)
        {
            TensorRef? weight = normHead.Inputs.Length > 1 ? Ref(normHead.Inputs[1]) : null;
            EmitRowNormKernel($"{normHead.Op.Name}+fused[{label}]", [normHead, .. group], shape, shapeValue.Symbol,
                inputs, preStmts: [], normExpr: headInput!, pass1Outputs: [], postStmts: statements,
                pass2Outputs: outs, weight, normHead.Op.Name == "RmsNorm" ? 0 : 1, normHead.Attributes.Get<float>("eps"));
            return;
        }

        EmitElementwiseKernel($"fused[{label}]", group, shape, shapeValue.Symbol, inputs, outs, statements);
    }

    /// <summary>
    /// Row normalization over the innermost axis fused with elementwise work before and/or after it (see
    /// <see cref="RowNormKernelSource"/> for the two-pass scheme), launched with one workgroup per row.
    /// </summary>
    private void EmitRowNormKernel(string name, IEnumerable<Node> nodes, long[] shape, string? symbol, List<TensorRef> inputs,
        List<string> preStmts, string normExpr, List<(TensorRef Ref, string Expr)> pass1Outputs, List<string> postStmts,
        List<(TensorRef Ref, string Expr)> pass2Outputs, TensorRef? w, int mode, float eps)
    {
        var source = new RowNormKernelSource
        {
            Shape = shape,
            Inputs = inputs,
            PreStatements = preStmts,
            NormExpression = normExpr,
            Pass1Outputs = pass1Outputs,
            PostStatements = postStmts,
            Pass2Outputs = pass2Outputs,
            HasWeight = w is not null,
            Mode = mode,
            Epsilon = eps,
            UseSubgroupOps = _opt.Has(DeviceFeatures.SubgroupOps),
        };
        var outputs = pass1Outputs.Concat(pass2Outputs).ToList();
        var kernel = _compiler.Compile("elementwise.slang", source.Build(), "main", _opt.Target, Macros(), _opt.Cache);

        long rowLength = shape[^1];
        long rowsPerSymbol = ElementwiseCodegen.Pitches(shape)[0] / rowLength;
        KernelArg rows = symbol is null ? new UIntArg((uint)(ElementwiseCodegen.ElementCount(shape) / rowLength)) : new SymbolArg(symbol, rowsPerSymbol);
        var weightOrFirstOutput = w ?? outputs[0].Ref;
        Emit(name, nodes,
            [Always(kernel, b => new Grid((uint)((symbol is null ? shape[0] : b[symbol]) * rowsPerSymbol)))],
            [.. inputs.Select(r => (KernelArg)Pointer(r)), Pointer(weightOrFirstOutput), .. outputs.Select(o => (KernelArg)Pointer(o.Ref)), rows],
            w is null ? inputs : [.. inputs, w], outputs.Select(o => o.Ref));
    }

    private void EmitElementwiseKernel(string name, IEnumerable<Node> nodes, long[] shape, string? symbol,
        List<TensorRef> inputs, List<(TensorRef Ref, string Expr)> outputs, List<string> stmts)
    {
        if (shape.Any(d => d == 0))
        {
            return;   // empty tensor: nothing to compute (and the stride maths would divide by zero)
        }

        var pitches = ElementwiseCodegen.Pitches(shape);
        long inner = shape.Length == 0 ? 1 : pitches[0];

        var sb = new StringBuilder();
        sb.AppendLine("#include \"math_ext.slang\"");
        sb.AppendLine("struct Params {");
        for (int i = 0; i < inputs.Count; i++)
        {
            sb.AppendLine($"  {ElementwiseCodegen.PointerType(inputs[i].Type)}* in{i};");
        }

        for (int i = 0; i < outputs.Count; i++)
        {
            sb.AppendLine($"  float* out{i};");
        }

        sb.AppendLine("  uint n;");
        sb.AppendLine("};");
        sb.AppendLine("[shader(\"compute\")] [numthreads(256, 1, 1)]");
        sb.AppendLine("void main(uint3 dtid : SV_DispatchThreadID, uniform Params p) {");
        sb.AppendLine("  uint idx = dtid.x; if (idx >= p.n) return;");
        ElementwiseCodegen.AppendCoordinates(sb, "  ", shape, pitches);
        for (int i = 0; i < inputs.Count; i++)
        {
            sb.AppendLine($"  float a{i} = {ElementwiseCodegen.Load(inputs[i], i, shape)};");
        }

        foreach (var statement in stmts)
        {
            sb.AppendLine("  " + statement);
        }

        for (int i = 0; i < outputs.Count; i++)
        {
            sb.AppendLine($"  p.out{i}[{ElementwiseCodegen.Offset(outputs[i].Ref, shape)}] = {outputs[i].Expr};");
        }

        sb.AppendLine("}");

        var kernel = _compiler.Compile("elementwise.slang", sb.ToString(), "main", _opt.Target, Macros(), _opt.Cache);
        KernelArg count = symbol is null ? new UIntArg((uint)ElementwiseCodegen.ElementCount(shape)) : new SymbolArg(symbol, inner);
        Emit(name, nodes,
            [Always(kernel, b => new Grid(Grid.Div((symbol is null ? shape[0] : b[symbol]) * inner, 256)))],
            [.. inputs.Select(r => (KernelArg)Pointer(r)), .. outputs.Select(o => (KernelArg)Pointer(o.Ref)), count],
            inputs, outputs.Select(o => o.Ref));
    }
}
