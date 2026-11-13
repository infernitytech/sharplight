using System.Globalization;
using System.Text;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

/// <summary>Lowerings for the general-purpose tensor ops (data movement, indexing, scatter, scans, top-k, ...).
/// Most are emitted as generated Slang kernels (<see cref="EmitGen"/>) that work on any element type.</summary>
public sealed partial class GraphLowerer
{
    private static string[] GeneralOpNames =>
        ["Expand", "Tile", "GatherAxis", "GatherElements", "GatherND", "ScatterElements", "ScatterND", "CumSum", "ArgReduce", "TopK",
         "GroupNorm", "ConvTranspose", "Trilu", "OneHot", "GridSample", "Fill"];
    private static readonly HashSet<string> s_generalOps = [.. GeneralOpNames];

    private static bool IsDataType(DataType t) => !t.IsQuantized && (t == DataType.F32 || t == DataType.F16 || t == DataType.BF16 || t == DataType.I32 || t == DataType.I64 || t == DataType.U32);
    private static bool IsIndexType(DataType t) => t == DataType.I32 || t == DataType.I64;

    private static OpSupport SupportsGeneral(Node node)
    {
        var a = node.Attributes;
        string op = node.Op.Name;
        foreach (var v in node.Inputs)
        {
            if (!IsDataType(v.Type))
            {
                return OpSupport.No($"{op} input type {v.Type}");
            }
        }

        switch (op)
        {
            case "Fill":
                return node.Output.Type == DataType.F32 || node.Output.Type == DataType.I32 ? OpSupport.Yes : OpSupport.No($"Fill of {node.Output.Type}");
            case "TopK" when node.Attributes.Has("k_symbol"):
                return OpSupport.No("TopK with a data-dependent k");
            case "GatherAxis":
            case "GatherElements":
            case "GatherND":
                if (!IsIndexType(node.Inputs[1].Type))
                {
                    return OpSupport.No($"{op} indices must be i32/i64");
                }

                break;
            case "ScatterElements":
            case "ScatterND":
                if (!IsIndexType(node.Inputs[1].Type))
                {
                    return OpSupport.No($"{op} indices must be i32/i64");
                }

                if (node.Inputs[0].Type != node.Inputs[2].Type)
                {
                    return OpSupport.No($"{op}: data and updates types differ");
                }

                if (a.Get("reduction", "none") != "none" && node.Inputs[0].Type != DataType.F32 && node.Inputs[0].Type != DataType.I32)
                {
                    return OpSupport.No($"{op} reduction on {node.Inputs[0].Type}");
                }

                if (node.Inputs[1].Shape.Symbol is not null && (op == "ScatterND" ? a.Get("batch_dims", 0L) == 0 : Axis0(a, node.Inputs[0].Rank)))
                {
                    return OpSupport.No($"{op} over a symbolic axis");
                }

                break;
            case "OneHot":
                if (!IsIndexType(node.Inputs[0].Type) && node.Inputs[0].Type != DataType.F32)
                {
                    return OpSupport.No("OneHot indices type");
                }

                break;
            case "CumSum":
            case "GroupNorm":
            case "ConvTranspose":
            case "GridSample":
            case "TopK":
            case "ArgReduce":
                if (node.Inputs[0].Type == DataType.I64 || node.Inputs[0].Type == DataType.U32)
                {
                    return OpSupport.No($"{op} on {node.Inputs[0].Type}");
                }

                break;
        }
        if (op == "ConvTranspose" && node.Inputs[1].Type == DataType.I32)
        {
            return OpSupport.No("ConvTranspose weight type");
        }

        if (op == "GridSample" && !(a.Get("mode", "linear") is "linear" or "nearest" or "cubic"))
        {
            return OpSupport.No("GridSample mode");
        }

        if (op == "GridSample" && !(a.Get("padding", "zeros") is "zeros" or "border" or "reflection"))
        {
            return OpSupport.No("GridSample padding");
        }

        return OpSupport.Yes;
    }

    private static int NormAxis(long axis, int rank) => (int)(axis < 0 ? axis + rank : axis);
    private static bool Axis0(Attributes a, int rank) => NormAxis(a.Get("axis", 0L), rank) == 0;

    private bool LowerGeneral(Node n)
    {
        switch (n.Op.Name)
        {
            case "Cast" when n.Output.Type != DataType.F32: LowerCastTyped(n); return true;
            case "Where" when n.Output.Type != DataType.F32: LowerWhereTyped(n); return true;
            case "Expand": { var x = Ref(n.Inputs[0]); EmitCopy(x, Produce(n.Output), "expand", [n]); return true; }
            case "Tile": LowerTile(n); return true;
            case "GatherAxis": LowerGatherAxis(n); return true;
            case "GatherElements": LowerGatherElements(n); return true;
            case "GatherND": LowerGatherND(n); return true;
            case "ScatterElements": LowerScatterElements(n); return true;
            case "ScatterND": LowerScatterND(n); return true;
            case "CumSum": LowerCumSum(n); return true;
            case "ArgReduce": LowerArgReduce(n); return true;
            case "TopK": LowerTopK(n); return true;
            case "GroupNorm": LowerGroupNorm(n); return true;
            case "ConvTranspose": LowerConvTranspose(n); return true;
            case "Trilu": LowerTrilu(n); return true;
            case "OneHot": LowerOneHot(n); return true;
            case "GridSample": LowerGridSample(n); return true;
            case "Fill": LowerFill(n); return true;
        }
        return false;
    }

    // ---- generated kernels ---------------------------------------------------------------------
    private static string CType(DataType t) =>
        t == DataType.F32 ? "float" : t == DataType.I32 ? "int" : t == DataType.U32 ? "uint" : t == DataType.I64 ? "int64_t"
        : t == DataType.F16 || t == DataType.BF16 || t == DataType.I16 ? "uint16_t"
        : t == DataType.U8 || t == DataType.I8 || t == DataType.Bool ? "uint8_t" : throw new NotSupportedException($"element type {t}");

    private static string RawType(DataType t) => t.ElementSize switch { 1 => "uint8_t", 2 => "uint16_t", 4 => "uint", 8 => "uint64_t", _ => throw new NotSupportedException() };

    private static string Dn(int k) => "d" + k;

    /// <summary>Address expression of <paramref name="r"/> given one coordinate expression per dim of r.</summary>
    private static string Lin(TensorRef r, IReadOnlyList<string> coords)
    {
        var terms = new List<string>();
        for (int k = 0; k < r.Rank; k++)
        {
            if (r.Shape[k] == 1 && !(k == 0 && r.Symbol is not null))
            {
                continue;
            }

            terms.Add(r.Strides[k] == 1 ? $"({coords[k]})" : $"({coords[k]}) * {r.Strides[k]}u");
        }
        return terms.Count == 0 ? "0" : string.Join(" + ", terms);
    }

    /// <summary>Address of <paramref name="r"/> at output coordinates d0.. of <paramref name="outShape"/> with trailing alignment and size-1 broadcasting.</summary>
    private static string BroadcastLin(TensorRef r, long[] outShape)
    {
        var coords = new List<string>();
        for (int k = 0; k < r.Rank; k++)
        {
            int ok = outShape.Length - r.Rank + k;
            coords.Add(ok < 0 || (r.Shape[k] == 1 && outShape[ok] != 1) ? "0" : Dn(ok));
        }
        return Lin(r, coords);
    }

    /// <summary>
    /// Emits a generated one-thread-per-element kernel over iteration space <paramref name="shape"/> (outermost dim possibly symbolic).
    /// The body can use idx, d0..dN-1, p.in{i} and p.out{i} (typed pointers, see <paramref name="inTypes"/>/<paramref name="outTypes"/>).
    /// </summary>
    private void EmitGen(string name, IEnumerable<Node> nodes, long[] shape, string? symbol, IReadOnlyList<TensorRef> ins, IReadOnlyList<TensorRef> outs, string body,
        string[]? inTypes = null, string[]? outTypes = null, bool outsAreReads = false)
    {
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
        for (int i = 0; i < ins.Count; i++)
        {
            sb.AppendLine($"  {(inTypes?[i] ?? CType(ins[i].Type))}* in{i};");
        }

        for (int i = 0; i < outs.Count; i++)
        {
            sb.AppendLine($"  {(outTypes?[i] ?? CType(outs[i].Type))}* out{i};");
        }

        sb.AppendLine("  uint n;\n};");
        sb.AppendLine("[shader(\"compute\")] [numthreads(256, 1, 1)]");
        sb.AppendLine("void main(uint3 dtid : SV_DispatchThreadID, uniform Params p) {");
        sb.AppendLine("  uint idx = dtid.x; if (idx >= p.n) return;");
        for (int k = 0; k < rank; k++)
        {
            sb.AppendLine(k == 0 ? $"  uint d0 = idx / {P[0]}u;" : $"  uint d{k} = (idx / {P[k]}u) % {shape[k]}u;");
        }

        sb.AppendLine(body);
        sb.AppendLine("}");
        var kernel = _compiler.Compile("gen.slang", sb.ToString(), "main", _opt.Target, Macros(), _opt.Cache);
        long total = shape.Aggregate(1L, (a, b) => a * b);
        KernelArg count = symbol is null ? new UIntArg((uint)total) : new SymbolArg(symbol, inner);
        Emit(name, nodes,
            [Always(kernel, b => new Grid(Grid.Div((symbol is null ? total : b[symbol] * inner), 256)))],
            [.. ins.Select(r => (KernelArg)Pointer(r)), .. outs.Select(o => (KernelArg)Pointer(o)), count],
            outsAreReads ? [.. ins, .. outs] : ins, outs);
    }

    /// <summary>Raw (bit-exact) strided copy for non-f32 types of equal element size; broadcasting like the f32 path.</summary>
    private void EmitRawCopy(TensorRef src, TensorRef dst, string name, IEnumerable<Node> nodes)
    {
        string t = RawType(src.Type);
        EmitGen(name, nodes, dst.Shape, dst.Symbol, [src], [dst],
            $"  p.out0[{Lin(dst, Enumerable.Range(0, dst.Rank).Select(Dn).ToArray())}] = p.in0[{BroadcastLin(src, dst.Shape)}];", [t], [t]);
    }

    private static string ToFloat(DataType t, string e) =>
        t == DataType.F32 ? e : t == DataType.F16 ? $"h2f({e})" : t == DataType.BF16 ? $"bf2f({e})" : $"float({e})";

    private static string FromFloat(DataType t, string e) =>
        t == DataType.F32 ? e : t == DataType.F16 ? $"f2h({e})" : t == DataType.BF16 ? $"f2bf({e})"
        : t == DataType.I64 ? $"int64_t({e})" : t == DataType.U32 ? $"uint({e})" : $"int({e})";

    private static bool IsFloatType(DataType t) => t == DataType.F32 || t == DataType.F16 || t == DataType.BF16;

    // ---- Cast / Where with non-f32 results -----------------------------------------------------
    private void LowerCastTyped(Node n)
    {
        var src = Ref(n.Inputs[0]);
        var dst = Produce(n.Output);
        string load = $"p.in0[{BroadcastLin(src, dst.Shape)}]";
        string v;
        if (IsFloatType(dst.Type) || IsFloatType(src.Type))
        {
            v = FromFloat(dst.Type, ToFloat(src.Type, load));
        }
        else
        {
            v = dst.Type == DataType.I64 ? $"int64_t({load})" : dst.Type == DataType.U32 ? $"uint({load})" : $"int({load})";
        }

        if (src.Type == dst.Type)
        {
            v = load;
        }

        EmitGen($"cast {src.Type}->{dst.Type}", [n], dst.Shape, dst.Symbol, [src], [dst], $"  p.out0[idx] = {v};");
    }

    private void LowerWhereTyped(Node n)
    {
        var c = Ref(n.Inputs[0]);
        var a = Ref(n.Inputs[1]);
        var b = Ref(n.Inputs[2]);
        var dst = Produce(n.Output);
        string cl = ToFloat(c.Type, $"p.in0[{BroadcastLin(c, dst.Shape)}]");
        EmitGen("where", [n], dst.Shape, dst.Symbol, [c, a, b], [dst],
            $"  p.out0[idx] = ({cl} != 0.0) ? p.in1[{BroadcastLin(a, dst.Shape)}] : p.in2[{BroadcastLin(b, dst.Shape)}];");
    }

    // ---- Tile / gathers --------------------------------------------------------------------------
    private void LowerTile(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var y = Produce(n.Output);
        var coords = Enumerable.Range(0, x.Rank).Select(k => k == 0 && x.Symbol is not null ? Dn(0) : $"({Dn(k)} % {x.Shape[k]}u)").ToArray();
        EmitGen("tile", [n], y.Shape, y.Symbol, [x], [y], $"  p.out0[idx] = p.in0[{Lin(x, coords)}];");
    }

    private void LowerGatherAxis(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var ix = Ref(n.Inputs[1]);
        var y = Produce(n.Output);
        int axis = NormAxis(n.Attributes.Get("axis", 0L), x.Rank);
        int q = ix.Rank;
        long A = x.Shape[axis];
        var icoords = Enumerable.Range(0, q).Select(j => Dn(axis + j)).ToArray();
        var xcoords = new string[x.Rank];
        for (int k = 0; k < x.Rank; k++)
        {
            xcoords[k] = k < axis ? Dn(k) : k == axis ? "uint(ii)" : Dn(axis + q + (k - axis - 1));
        }

        string body = $"  int ii = int(p.in1[{Lin(ix, icoords)}]); if (ii < 0) ii += {A}; ii = clamp(ii, 0, {A - 1});\n  p.out0[idx] = p.in0[{Lin(x, xcoords)}];";
        EmitGen("gather_axis", [n], y.Shape, y.Symbol, [x, ix], [y], body);
    }

    private void LowerGatherElements(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var ix = Ref(n.Inputs[1]);
        var y = Produce(n.Output);
        int axis = NormAxis(n.Attributes.Get("axis", 0L), x.Rank);
        long A = x.Shape[axis];
        var d = Enumerable.Range(0, x.Rank).Select(Dn).ToArray();
        var xcoords = d.Select((c, k) => k == axis ? "uint(ii)" : c).ToArray();
        string body = $"  int ii = int(p.in1[{Lin(ix, d)}]); if (ii < 0) ii += {A}; ii = clamp(ii, 0, {A - 1});\n  p.out0[idx] = p.in0[{Lin(x, xcoords)}];";
        EmitGen("gather_elements", [n], y.Shape, y.Symbol, [x, ix], [y], body);
    }

    private void LowerGatherND(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var ix = Ref(n.Inputs[1]);
        var y = Produce(n.Output);
        int b = (int)n.Attributes.Get("batch_dims", 0L);
        int q = (int)ix.Shape[^1];
        int iq = ix.Rank - 1;
        var sb = new StringBuilder();
        var xcoords = new string[x.Rank];
        for (int j = 0; j < x.Rank; j++)
        {
            if (j < b)
            {
                xcoords[j] = Dn(j);
            }
            else if (j < b + q)
            {
                var ic = Enumerable.Range(0, iq).Select(Dn).Append((j - b).ToString()).ToArray();
                sb.AppendLine($"  int i{j} = int(p.in1[{Lin(ix, ic)}]); if (i{j} < 0) i{j} += {x.Shape[j]}; i{j} = clamp(i{j}, 0, {x.Shape[j] - 1});");
                xcoords[j] = $"uint(i{j})";
            }
            else
            {
                xcoords[j] = Dn(iq + (j - b - q));
            }
        }
        sb.AppendLine($"  p.out0[idx] = p.in0[{Lin(x, xcoords)}];");
        EmitGen("gather_nd", [n], y.Shape, y.Symbol, [x, ix], [y], sb.ToString());
    }

    // ---- scatter (deterministic: sequential along the scatter axis) -------------------------------
    private static string ScatterOp(string reduction, string o, string u) => reduction switch
    {
        "add" => $"({o} + {u})",
        "mul" => $"({o} * {u})",
        "max" => $"max({o}, {u})",
        "min" => $"min({o}, {u})",
        _ => u,
    };

    private void LowerScatterElements(Node n)
    {
        var data = Ref(n.Inputs[0]);
        var ix = Ref(n.Inputs[1]);
        var upd = Ref(n.Inputs[2]);
        var y = Produce(n.Output);
        EmitCopy(data, y, "scatter-copy", [n]);
        int axis = NormAxis(n.Attributes.Get("axis", 0L), data.Rank);
        long A = data.Shape[axis];
        var iter = (long[])ix.Shape.Clone();
        long len = iter[axis];
        iter[axis] = 1;
        var d = Enumerable.Range(0, ix.Rank).Select(Dn).ToArray();
        var cj = d.Select((c, k) => k == axis ? "j" : c).ToArray();
        var ycoords = d.Select((c, k) => k == axis ? "uint(ii)" : c).ToArray();
        string red = n.Attributes.Get("reduction", "none");
        string body = $"  for (uint j = 0; j < {len}u; j++) {{\n" +
            $"    int ii = int(p.in0[{Lin(ix, cj)}]); if (ii < 0) ii += {A}; if (ii < 0 || ii >= {A}) continue;\n" +
            $"    uint yo = {Lin(y, ycoords)};\n    p.out0[yo] = {ScatterOp(red, "p.out0[yo]", $"p.in1[{Lin(upd, cj)}]")};\n  }}";
        EmitGen("scatter_elements", [n], iter, ix.Symbol, [ix, upd], [y], body, outsAreReads: true);
    }

    private void LowerScatterND(Node n)
    {
        if (TryLowerScatterNDFast(n))
        {
            return;
        }

        var data = Ref(n.Inputs[0]);
        var ix = Ref(n.Inputs[1]);
        var upd = Ref(n.Inputs[2]);
        var y = Produce(n.Output);
        EmitCopy(data, y, "scatter-copy", [n]);
        int b = (int)n.Attributes.Get("batch_dims", 0L);
        int q = (int)ix.Shape[^1];
        int iq = ix.Rank - 1;
        var iter = ix.Shape.Take(b).Concat(data.Shape.Skip(b + q)).ToArray();
        string? sym = b > 0 ? ix.Symbol : null;
        long M = 1; for (int k = b; k < iq; k++)
        {
            M *= ix.Shape[k];
        }

        int tailN = data.Rank - b - q;
        // iteration coords: d0..d{b-1} batch, d{b}.. tail
        var sb = new StringBuilder();
        sb.AppendLine($"  for (uint m = 0; m < {M}u; m++) {{");
        // decompose m over idx dims b..iq-1
        var mc = new List<string>();
        long rem = M;
        for (int k = b; k < iq; k++) { rem /= ix.Shape[k]; mc.Add($"((m / {rem}u) % {ix.Shape[k]}u)"); }
        var e = Enumerable.Range(0, b).Select(Dn).Concat(mc).ToArray();
        var ycoords = new string[data.Rank];
        for (int j = 0; j < data.Rank; j++)
        {
            if (j < b)
            {
                ycoords[j] = Dn(j);
            }
            else if (j < b + q)
            {
                var ic = e.Append((j - b).ToString()).ToArray();
                sb.AppendLine($"    int i{j} = int(p.in0[{Lin(ix, ic)}]); if (i{j} < 0) i{j} += {data.Shape[j]}; if (i{j} < 0 || i{j} >= {data.Shape[j]}) continue;");
                ycoords[j] = $"uint(i{j})";
            }
            else
            {
                ycoords[j] = Dn(b + (j - b - q));
            }
        }
        var ucoords = e.Concat(Enumerable.Range(0, tailN).Select(t => Dn(b + t))).ToArray();
        string red = n.Attributes.Get("reduction", "none");
        sb.AppendLine($"    uint yo = {Lin(y, ycoords)};\n    p.out0[yo] = {ScatterOp(red, "p.out0[yo]", $"p.in1[{Lin(upd, ucoords)}]")};\n  }}");
        EmitGen("scatter_nd", [n], iter, sym, [ix, upd], [y], sb.ToString(), outsAreReads: true);
    }

    // ---- scans / arg-reductions / top-k ------------------------------------------------------------
    private void LowerCumSum(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "cumsum");
        var y = Produce(n.Output);
        int axis = NormAxis(n.Attributes.Get("axis", 0L), x.Rank);
        bool excl = n.Attributes.Get("exclusive", false), rev = n.Attributes.Get("reverse", false);
        var iter = (long[])x.Shape.Clone();
        long len = iter[axis];
        iter[axis] = 1;
        var d = Enumerable.Range(0, x.Rank).Select(Dn).ToArray();
        var cj = d.Select((c, k) => k == axis ? "j" : c).ToArray();
        string body = $"  float acc = 0.0;\n  for (uint i = 0; i < {len}u; i++) {{\n    uint j = {(rev ? $"{len - 1}u - i" : "i")};\n" +
            $"    float v = p.in0[{Lin(x, cj)}];\n" +
            (excl ? $"    p.out0[{Lin(y, cj)}] = acc; acc += v;\n" : $"    acc += v; p.out0[{Lin(y, cj)}] = acc;\n") + "  }";
        EmitGen("cumsum", [n], iter, x.Symbol, [x], [y], body);
    }

    private void LowerArgReduce(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "argreduce");
        var y = Produce(n.Output);
        int axis = NormAxis(n.Attributes.Get("axis", 0L), x.Rank);
        bool max = n.Attributes.Get("mode", "max") == "max", last = n.Attributes.Get("select_last_index", false);
        var iter = (long[])x.Shape.Clone();
        long len = iter[axis];
        iter[axis] = 1;
        var d = Enumerable.Range(0, x.Rank).Select(Dn).ToArray();
        var cj = d.Select((c, k) => k == axis ? "j" : c).ToArray();
        string cmp = max ? ">" : "<";
        string body = $"  float best = 0.0; int bi = -1;\n  for (uint j = 0; j < {len}u; j++) {{\n    float v = p.in0[{Lin(x, cj)}];\n" +
            $"    if (bi < 0 || v {cmp} best{(last ? $" || v == best" : "")}) {{ best = v; bi = int(j); }}\n  }}\n  p.out0[idx] = bi;";
        EmitGen("argreduce", [n], iter, x.Symbol, [x], [y], body);
    }

    /// <summary>y[i, ...] = start + i·delta along the (possibly symbolic) outer axis.</summary>
    private void LowerFill(Node n)
    {
        var y = Produce(n.Output);
        float start = n.Attributes.Get("start", 0f);
        float delta = n.Attributes.Get("delta", 0f);
        string value = $"float({start.ToString("R", CultureInfo.InvariantCulture)}) + float(d0) * float({delta.ToString("R", CultureInfo.InvariantCulture)})";
        string store = y.Type == DataType.I32 ? $"int({value})" : $"float({value})";
        EmitGen($"fill {n.Output.Name}", [n], y.Shape, y.Symbol, [], [y], $"  p.out0[idx] = {store};");
    }

    /// <summary>Serial top-k (one thread per row, O(k * n)): fallback when k exceeds the shared-memory sort of <see cref="LowerTopK"/>.</summary>
    private void LowerTopKSerial(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "topk");
        var vals = Produce(n.Outputs[0]);
        var idxs = Produce(n.Outputs[1]);
        int axis = NormAxis(n.Attributes.Get("axis", -1L), x.Rank);
        long K = n.Attributes.Get<long>("k");
        bool largest = n.Attributes.Get("largest", true);
        var iter = (long[])x.Shape.Clone();
        long len = iter[axis];
        iter[axis] = 1;
        var d = Enumerable.Range(0, x.Rank).Select(Dn).ToArray();
        var cj = d.Select((c, k) => k == axis ? "j" : c).ToArray();
        var ck = d.Select((c, k) => k == axis ? "r" : c).ToArray();
        string beforeV = largest ? "v < pv" : "v > pv", better = largest ? "v > bv" : "v < bv";
        string body = $"  float pv = 0.0; int pi = -1;\n  for (uint r = 0; r < {K}u; r++) {{\n    float bv = 0.0; int bi = -1;\n" +
            $"    for (uint j = 0; j < {len}u; j++) {{\n      float v = p.in0[{Lin(x, cj)}];\n" +
            $"      bool after = (r == 0) || ({beforeV}) || (v == pv && int(j) > pi);\n      if (!after) continue;\n" +
            $"      if (bi < 0 || {better}) {{ bv = v; bi = int(j); }}\n    }}\n" +
            $"    p.out0[{Lin(vals, ck)}] = bv; p.out1[{Lin(idxs, ck)}] = bi; pv = bv; pi = bi;\n  }}";
        EmitGen("topk", [n], iter, x.Symbol, [x], [vals, idxs], body);
    }

    // ---- misc generated ops ---------------------------------------------------------------------
    private void LowerTrilu(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var y = Produce(n.Output);
        bool upper = n.Attributes.Get("upper", true);
        long k = n.Attributes.Get("k", 0L);
        int r = x.Rank;
        string cond = upper ? $"(int({Dn(r - 1)}) - int({Dn(r - 2)}) >= {k})" : $"(int({Dn(r - 1)}) - int({Dn(r - 2)}) <= {k})";
        string ct = CType(x.Type);
        EmitGen("trilu", [n], y.Shape, y.Symbol, [x], [y], $"  p.out0[idx] = {cond} ? p.in0[{BroadcastLin(x, y.Shape)}] : {ct}(0);");
    }

    private void LowerOneHot(Node n)
    {
        var x = Ref(n.Inputs[0]);
        var y = Produce(n.Output);
        int axis = (int)n.Attributes.Get("axis", -1L); if (axis < 0)
        {
            axis += x.Rank + 1;
        }

        long depth = n.Attributes.Get<long>("depth");
        var coords = Enumerable.Range(0, y.Rank).Where(k => k != axis).Select(Dn).ToArray();
        string on = F(n.Attributes.Get("on", 1f)), off = F(n.Attributes.Get("off", 0f));
        string body = $"  int ii = int(p.in0[{Lin(x, coords)}]); if (ii < 0) ii += {depth};\n  p.out0[idx] = (uint(ii) == {Dn(axis)}) ? {on} : {off};";
        EmitGen("onehot", [n], y.Shape, y.Symbol, [x], [y], body);
    }

    // ---- library kernels ----------------------------------------------------------------------------
    private void LowerGroupNorm(Node n)
    {
        var x = AsDenseF32(Ref(n.Inputs[0]), "groupnorm");
        var sc = n.Inputs.Length > 1 ? AsDenseF32(Ref(n.Inputs[1]), "gn-scale") : null;
        var bi = n.Inputs.Length > 2 ? AsDenseF32(Ref(n.Inputs[2]), "gn-bias") : null;
        var y = Produce(n.Output);
        long C = x.Shape[1], S = 1; for (int k = 2; k < x.Rank; k++)
        {
            S *= x.Shape[k];
        }

        long G = n.Attributes.Get("groups", 1L);
        if (TryLowerGroupNormSplit(n, x, sc, bi, y, C, S, G))
        {
            return;
        }

        var k2 = Lib("general.slang", "groupnorm", Macros(("C", C), ("S", S), ("G", G), ("EPS", n.Attributes.Get("eps", 1e-5f)), ("HAS_SCALE", sc is not null), ("HAS_BIAS", bi is not null)));
        var tx = x;
        var sp = sc ?? x;
        var bp = bi ?? x;
        Emit($"groupnorm {n.Output.Name}", [n], [Always(k2, b => new Grid((uint)(Rows(b, tx) * G)))],
            [Pointer(x), Pointer(sp), Pointer(bp), Pointer(y), TokenArg(x)],
            [.. new[] { x, sc, bi }.Where(t => t is not null).Select(t => t!)], [y]);
    }

    private void LowerConvTranspose(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "convt");
        var w = Ref(n.Inputs[1]);
        w = EnsureDense(w, "convt-w");

        var bias = n.Inputs.Length > 2 ? AsDenseF32(Ref(n.Inputs[2]), "convt-bias") : null;
        var a = n.Attributes;
        var st = a.Get("strides", new long[] { 1, 1 });
        var pd = a.Get("pads", new long[] { 0, 0, 0, 0 });
        var dl = a.Get("dilations", new long[] { 1, 1 });
        long G = a.Get("groups", 1L);
        long C = x.Shape[1];
        long OG = w.Shape[1];
        var y = Produce(n.Output);
        long O = OG * G;
        int fmt = WeightFormat(w.Type) ?? throw new NotSupportedException($"ConvTranspose weight type {w.Type}");
        var k = Lib("general.slang", "conv_transpose", Macros(("C", C), ("H", x.Shape[2]), ("W", x.Shape[3]), ("O", O), ("HO", y.Shape[2]), ("WO", y.Shape[3]),
            ("CG", C / G), ("OG", OG), ("G", G), ("KH", w.Shape[2]), ("KW", w.Shape[3]), ("SH", st[0]), ("SW", st[1]), ("PH", pd[0]), ("PW", pd[1]), ("DH", dl[0]), ("DW", dl[1]),
            ("XS_N", x.Strides[0]), ("XS_C", x.Strides[1]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]), ("WFMT", fmt), ("HAS_BIAS", bias is not null)));
        var tx = x;
        var bp = bias ?? x;
        Emit($"conv_transpose {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * O * y.Shape[2] * y.Shape[3], 256)))],
            [Pointer(x), Pointer(w), Pointer(bp), Pointer(y), CountArg(x, O * y.Shape[2] * y.Shape[3])],
            bias is null ? [x, w] : [x, w, bias], [y]);
    }

    private void LowerGridSample(Node n)
    {
        var x = AsF32(Ref(n.Inputs[0]), "gridsample");
        var g = AsDenseF32(Ref(n.Inputs[1]), "gridsample-grid");
        var y = Produce(n.Output);
        var a = n.Attributes;
        long C = x.Shape[1];
        long HO = y.Shape[2];
        long WO = y.Shape[3];
        int mode = a.Get("mode", "linear") switch { "linear" => 0, "nearest" => 1, _ => 2 };
        int pad = a.Get("padding", "zeros") switch { "zeros" => 0, "border" => 1, _ => 2 };
        var k = Lib("general.slang", "grid_sample", Macros(("C", C), ("H", x.Shape[2]), ("W", x.Shape[3]), ("HO", HO), ("WO", WO),
            ("XS_N", x.Strides[0]), ("XS_C", x.Strides[1]), ("XS_H", x.Strides[2]), ("XS_W", x.Strides[3]), ("MODE", mode), ("PAD", pad), ("ALIGN", a.Get("align_corners", false))));
        var tx = x;
        Emit($"grid_sample {n.Output.Name}", [n], [Always(k, b => new Grid(Grid.Div(Rows(b, tx) * C * HO * WO, 256)))],
            [Pointer(x), Pointer(g), Pointer(y), CountArg(x, C * HO * WO)], [x, g], [y]);
    }
}
