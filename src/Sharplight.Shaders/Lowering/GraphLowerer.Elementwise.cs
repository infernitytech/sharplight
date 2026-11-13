using Sharplight.Runtime;
using System.Text;
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

    private void LowerElementwise(List<Node> group, List<Value> outputs, Node? normTail = null, Node? normHead = null)
    {
        var shapeV = group[0].Output.Shape;
        var inputs = new List<TensorRef>();
        var inputIndex = new Dictionary<Value, int>();
        var produced = group.Select(g => g.Output).ToHashSet();
        var stmts = new List<string>();
        var name = new Dictionary<Value, string>();

        string Operand(Value v)
        {
            if (name.TryGetValue(v, out var s))
            {
                return s;
            }

            if (!inputIndex.TryGetValue(v, out int idx))
            {
                idx = inputs.Count;
                inputs.Add(Ref(v));
                inputIndex[v] = idx;
            }
            return name[v] = "a" + idx;
        }

        string? headInput = null;
        if (normHead is not null)
        {
            headInput = Operand(normHead.Inputs[0]);
            name[normHead.Output] = "nrm";
        }
        foreach (var node in group)
        {
            var args = node.Inputs.Select(Operand).ToArray();
            string expr = OpExpr(node, args);
            var vn = "v" + stmts.Count;
            stmts.Add($"float {vn} = {expr};");
            name[node.Output] = vn;
        }

        var outs = outputs.Select(o => (Produce(o), name[o])).ToList();
        var shape = BoundShape(shapeV);
        var label = string.Join(",", group.Select(g => g.Op.Name));
        if (normTail is not null)
        {
            TensorRef? w = normTail.Inputs.Length > 1 ? Ref(normTail.Inputs[1]) : null;
            EmitRowNormKernel($"fused[{label}]+{normTail.Op.Name}", [.. group, normTail], shape, shapeV.Symbol,
                inputs, preStmts: stmts, normExpr: name[normTail.Inputs[0]], pass1Outputs: outs, postStmts: [],
                pass2Outputs: [(Produce(normTail.Output), "nrm")], w, normTail.Op.Name == "RmsNorm" ? 0 : 1, normTail.Attributes.Get<float>("eps"));
            return;
        }
        if (normHead is not null)
        {
            TensorRef? w = normHead.Inputs.Length > 1 ? Ref(normHead.Inputs[1]) : null;
            EmitRowNormKernel($"{normHead.Op.Name}+fused[{label}]", [normHead, .. group], shape, shapeV.Symbol,
                inputs, preStmts: [], normExpr: headInput!, pass1Outputs: [], postStmts: stmts,
                pass2Outputs: outs, w, normHead.Op.Name == "RmsNorm" ? 0 : 1, normHead.Attributes.Get<float>("eps"));
            return;
        }
        EmitElementwiseKernel($"fused[{string.Join(",", group.Select(g => g.Op.Name))}]", group, shape, shapeV.Symbol, inputs, outs, stmts);
    }

    /// <summary>
    /// Row normalization over the innermost axis fused with elementwise work before it (computing the normalized
    /// value) and/or after it (consuming <c>nrm</c>), one workgroup per row. Pass 1 evaluates the pre statements,
    /// writes <paramref name="pass1Outputs"/> and reduces the sum of squares; pass 2 re-evaluates the pre statements
    /// (cheaper than keeping a row in shared memory), forms <c>nrm</c>, runs the post statements and writes
    /// <paramref name="pass2Outputs"/>.
    /// </summary>
    private void EmitRowNormKernel(string name, IEnumerable<Node> nodes, long[] shape, string? symbol, List<TensorRef> inputs,
        List<string> preStmts, string normExpr, List<(TensorRef Ref, string Expr)> pass1Outputs, List<string> postStmts,
        List<(TensorRef Ref, string Expr)> pass2Outputs, TensorRef? w, int mode, float eps)
    {
        int rank = shape.Length;
        var P = new long[rank];
        long acc = 1;
        for (int k = rank - 1; k >= 0; k--)
        {
            P[k] = acc;
            acc *= shape[k];
        }
        long D = shape[^1];
        long inner = rank == 0 ? 1 : P[0];
        const int threads = 256;
        var outputs = pass1Outputs.Concat(pass2Outputs).ToList();

        var sb = new StringBuilder();
        sb.AppendLine("#include \"math_ext.slang\"");
        sb.AppendLine("struct Params {");
        for (int i = 0; i < inputs.Count; i++)
        {
            sb.AppendLine($"  {ElemPtrType(inputs[i].Type)}* in{i};");
        }

        sb.AppendLine("  float* w;");
        for (int i = 0; i < outputs.Count; i++)
        {
            sb.AppendLine($"  float* out{i};");
        }

        sb.AppendLine("  uint rows;");
        sb.AppendLine("};");
        sb.AppendLine($"groupshared float red[{threads}];");
        sb.AppendLine($"[shader(\"compute\")] [numthreads({threads}, 1, 1)]");
        sb.AppendLine("void main(uint3 gid : SV_GroupID, uint3 lt : SV_GroupThreadID, uniform Params p) {");
        sb.AppendLine("  uint row = gid.x; if (row >= p.rows) return;");
        // Fixed trip count per thread. Short rows (decode: one workgroup per row, latency bound) are fully
        // unrolled with every load issued before any store: inputs and outputs are plain pointers that may alias, so
        // an interleaved load/store loop would serialize one memory round trip per element. The loaded values stay in
        // registers for the second pass.
        long iters = (D + threads - 1) / threads;
        bool unroll = iters <= 32;
        void Indices(string indent)
        {
            sb.AppendLine($"{indent}uint idx = row * {D}u + i;");
            for (int k = 0; k < rank; k++)
            {
                sb.AppendLine(k == 0 ? $"{indent}uint d0 = idx / {P[0]}u;" : $"{indent}uint d{k} = (idx / {P[k]}u) % {shape[k]}u;");
            }
        }
        void Loop(string body)
        {
            sb.AppendLine($"  [ForceUnroll] for (uint j = 0; j < {iters}u; j++) {{");
            sb.AppendLine($"    uint i = lt.x + j * {threads}u;");
            if (D % threads != 0)
            {
                sb.AppendLine($"    if (i < {D}u) {{");
            }

            sb.Append(body);
            if (D % threads != 0)
            {
                sb.AppendLine("    }");
            }

            sb.AppendLine("  }");
        }
        void Body(bool second)
        {
            if (unroll)
            {
                var inner = new StringBuilder();
                var save = sb;
                sb = inner;
                Indices("    ");
                for (int i = 0; i < inputs.Count; i++)
                {
                    sb.AppendLine($"    float a{i} = ar{i}[j];");
                }

                foreach (var st in preStmts)
                {
                    sb.AppendLine("    " + st);
                }

                if (!second)
                {
                    for (int i = 0; i < pass1Outputs.Count; i++)
                    {
                        sb.AppendLine($"    p.out{i}[{ElemOffset(pass1Outputs[i].Ref, shape)}] = {pass1Outputs[i].Expr};");
                    }

                    sb.AppendLine($"    ss += {normExpr} * {normExpr};");
                }
                else
                {
                    sb.AppendLine($"    float nrm = {normExpr} * scale{(w is null ? "" : " * wr[j]")};");
                    foreach (var st in postStmts)
                    {
                        sb.AppendLine("    " + st);
                    }

                    for (int i = 0; i < pass2Outputs.Count; i++)
                    {
                        sb.AppendLine($"    p.out{pass1Outputs.Count + i}[{ElemOffset(pass2Outputs[i].Ref, shape)}] = {pass2Outputs[i].Expr};");
                    }
                }
                sb = save;
                if (!second)
                {
                    for (int i = 0; i < inputs.Count; i++)
                    {
                        sb.AppendLine($"  float ar{i}[{iters}];");
                    }

                    var loads = new StringBuilder();
                    var save2 = sb;
                    sb = loads;
                    Indices("    ");
                    for (int i = 0; i < inputs.Count; i++)
                    {
                        sb.AppendLine($"    ar{i}[j] = {ElemLoad(inputs[i], i, shape)};");
                    }

                    if (w is not null)
                    {
                        sb.AppendLine("    wr[j] = p.w[i];");
                    }

                    sb = save2;
                    if (w is not null)
                    {
                        sb.AppendLine($"  float wr[{iters}];");
                    }

                    Loop(loads.ToString());
                }
                Loop(inner.ToString());
                return;
            }
            sb.AppendLine($"  for (uint i = lt.x; i < {D}u; i += {threads}u) {{");
            Indices("    ");
            for (int i = 0; i < inputs.Count; i++)
            {
                sb.AppendLine($"    float a{i} = {ElemLoad(inputs[i], i, shape)};");
            }

            foreach (var st in preStmts)
            {
                sb.AppendLine("    " + st);
            }

            if (!second)
            {
                for (int i = 0; i < pass1Outputs.Count; i++)
                {
                    sb.AppendLine($"    p.out{i}[{ElemOffset(pass1Outputs[i].Ref, shape)}] = {pass1Outputs[i].Expr};");
                }

                sb.AppendLine($"    ss += {normExpr} * {normExpr};");
            }
            else
            {
                sb.AppendLine($"    float nrm = {normExpr} * scale{(w is null ? "" : " * p.w[i]")};");
                foreach (var st in postStmts)
                {
                    sb.AppendLine("    " + st);
                }

                for (int i = 0; i < pass2Outputs.Count; i++)
                {
                    sb.AppendLine($"    p.out{pass1Outputs.Count + i}[{ElemOffset(pass2Outputs[i].Ref, shape)}] = {pass2Outputs[i].Expr};");
                }
            }
            sb.AppendLine("  }");
        }
        sb.AppendLine("  float ss = 0;");
        Body(false);
        if (_opt.Has(DeviceFeatures.SubgroupOps))
        {
            // Subgroup sums, then one barrier and a short serial sum over the subgroups.
            sb.AppendLine("  float ws = WaveActiveSum(ss);");
            sb.AppendLine("  uint lanes = WaveGetLaneCount();");
            sb.AppendLine("  if (WaveIsFirstLane()) red[lt.x / lanes] = ws;");
            sb.AppendLine("  GroupMemoryBarrierWithGroupSync();");
            sb.AppendLine($"  float tot = 0; for (uint k = 0; k < {threads}u / lanes; k++) tot += red[k];");
        }
        else
        {
            sb.AppendLine("  red[lt.x] = ss;");
            sb.AppendLine("  GroupMemoryBarrierWithGroupSync();");
            sb.AppendLine($"  for (uint s = {threads / 2}u; s > 0; s >>= 1) {{ if (lt.x < s) red[lt.x] += red[lt.x + s]; GroupMemoryBarrierWithGroupSync(); }}");
            sb.AppendLine("  float tot = red[0];");
        }
        sb.AppendLine(mode == 0 ? $"  float scale = rsqrt(tot / {F(D)} + {F(eps)});" : $"  float scale = rsqrt(tot + {F(eps)});");
        Body(true);
        sb.AppendLine("}");

        var kernel = _compiler.Compile("elementwise.slang", sb.ToString(), "main", _opt.Target, Macros(), _opt.Cache);
        long rowsPerSym = inner / D;
        KernelArg rows = symbol is null ? new UIntArg((uint)(shape.Aggregate(1L, (a, b) => a * b) / D)) : new SymbolArg(symbol, rowsPerSym);
        var wArg = w ?? outputs[0].Ref;
        Emit(name, nodes,
            [Always(kernel, b => new Grid((uint)((symbol is null ? shape[0] : b[symbol]) * rowsPerSym)))],
            [.. inputs.Select(r => (KernelArg)Pointer(r)), Pointer(wArg),
             .. outputs.Select(o => (KernelArg)Pointer(o.Ref)), rows],
            w is null ? inputs : [.. inputs, w], outputs.Select(o => o.Ref));
    }

    private static string ElemOffset(TensorRef r, long[] shape)
    {
        int rank = shape.Length;
        var terms = new List<string>();
        for (int k = 0; k < rank; k++)
        {
            int rk = r.Rank - rank + k;
            if (rk < 0)
            {
                continue;
            }

            long st = r.Shape[rk] == 1 && shape[k] != 1 ? 0 : r.Strides[rk];
            if (st == 0 || (shape[k] == 1))
            {
                continue;
            }

            terms.Add(st == 1 ? $"d{k}" : $"d{k} * {st}u");
        }
        return terms.Count == 0 ? "0" : string.Join(" + ", terms);
    }

    private static string ElemLoad(TensorRef r, int i, long[] shape) => r.Type == DataType.F32 ? $"p.in{i}[{ElemOffset(r, shape)}]"
        : r.Type == DataType.F16 ? $"h2f(p.in{i}[{ElemOffset(r, shape)}])"
        : r.Type == DataType.BF16 ? $"bf2f(p.in{i}[{ElemOffset(r, shape)}])"
        : r.Type == DataType.I32 || r.Type == DataType.I64 || r.Type == DataType.U32 ? $"float(p.in{i}[{ElemOffset(r, shape)}])"
        : throw new NotSupportedException($"elementwise input type {r.Type}");

    private static string ElemPtrType(DataType t) => t == DataType.F32 ? "float" : t == DataType.I32 ? "int" : t == DataType.I64 ? "int64_t" : t == DataType.U32 ? "uint" : "uint16_t";

    private void EmitElementwiseKernel(string name, IEnumerable<Node> nodes, long[] shape, string? symbol,
        List<TensorRef> inputs, List<(TensorRef Ref, string Expr)> outputs, List<string> stmts)
    {
        if (shape.Any(d => d == 0))
        {
            return;   // empty tensor: nothing to compute (and the stride maths would divide by zero)
        }

        int rank = shape.Length;
        var P = new long[rank];
        long acc = 1;
        for (int k = rank - 1; k >= 0; k--)
        {
            P[k] = acc;
            acc *= shape[k];
        }
        long inner = rank == 0 ? 1 : P[0];

        var sb = new StringBuilder();
        sb.AppendLine("#include \"math_ext.slang\"");
        sb.AppendLine("struct Params {");
        for (int i = 0; i < inputs.Count; i++)
        {
            sb.AppendLine($"  {PtrType(inputs[i].Type)}* in{i};");
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
        for (int k = 0; k < rank; k++)
        {
            sb.AppendLine(k == 0 ? $"  uint d0 = idx / {P[0]}u;" : $"  uint d{k} = (idx / {P[k]}u) % {shape[k]}u;");
        }

        for (int i = 0; i < inputs.Count; i++)
        {
            sb.AppendLine($"  float a{i} = {Load(inputs[i], i)};");
        }

        foreach (var s in stmts)
        {
            sb.AppendLine("  " + s);
        }

        for (int i = 0; i < outputs.Count; i++)
        {
            sb.AppendLine($"  p.out{i}[{Offset(outputs[i].Ref)}] = {outputs[i].Expr};");
        }

        sb.AppendLine("}");

        var kernel = _compiler.Compile("elementwise.slang", sb.ToString(), "main", _opt.Target, Macros(), _opt.Cache);
        long total = shape.Aggregate(1L, (a, b) => a * b);
        KernelArg count = symbol is null ? new UIntArg((uint)total) : new SymbolArg(symbol, inner);
        Emit(name, nodes,
            [Always(kernel, b => new Grid(Grid.Div((symbol is null ? shape[0] : b[symbol]) * inner, 256)))],
            [.. inputs.Select(r => (KernelArg)Pointer(r)), .. outputs.Select(o => (KernelArg)Pointer(o.Ref)), count],
            inputs, outputs.Select(o => o.Ref));

        string Offset(TensorRef r)
        {
            var terms = new List<string>();
            for (int k = 0; k < rank; k++)
            {
                int rk = r.Rank - rank + k;
                if (rk < 0)
                {
                    continue;
                }

                long st = r.Shape[rk] == 1 && shape[k] != 1 ? 0 : r.Strides[rk];
                if (st == 0 || (shape[k] == 1))
                {
                    continue;
                }

                terms.Add(st == 1 ? $"d{k}" : $"d{k} * {st}u");
            }
            return terms.Count == 0 ? "0" : string.Join(" + ", terms);
        }

        string Load(TensorRef r, int i) => r.Type == DataType.F32 ? $"p.in{i}[{Offset(r)}]"
            : r.Type == DataType.F16 ? $"h2f(p.in{i}[{Offset(r)}])"
            : r.Type == DataType.BF16 ? $"bf2f(p.in{i}[{Offset(r)}])"
            : r.Type == DataType.I32 || r.Type == DataType.I64 || r.Type == DataType.U32 ? $"float(p.in{i}[{Offset(r)}])"
            : throw new NotSupportedException($"elementwise input type {r.Type}");

        static string PtrType(DataType t) => t == DataType.F32 ? "float" : t == DataType.I32 ? "int" : t == DataType.I64 ? "int64_t" : t == DataType.U32 ? "uint" : "uint16_t";
    }
}
