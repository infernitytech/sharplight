using System.Text;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Generates the Slang source of a row-normalization kernel fused with elementwise work before it (computing the
/// normalized value) and/or after it (consuming <c>nrm</c>), one workgroup per row. Pass 1 evaluates the pre statements,
/// writes the pass-1 outputs and reduces the sum of squares; pass 2 re-evaluates the pre statements (cheaper than keeping
/// a row in shared memory), forms <c>nrm</c>, runs the post statements and writes the pass-2 outputs.
/// </summary>
internal sealed class RowNormKernelSource
{
    private const int Threads = 256;

    /// <summary>Unrolling (all loads issued first) is used up to this many elements per thread.</summary>
    private const long MaxUnrolledIterations = 32;

    private readonly StringBuilder _sb = new();
    private long[] _pitches = [];
    private long _rowLength;
    private long _iterations;
    private bool _unroll;

    /// <summary>Broadcast shape of the elementwise work; the last axis is the normalized one.</summary>
    public required long[] Shape { get; init; }

    /// <summary>Tensors read by the statements, as <c>a0 .. aN</c>.</summary>
    public required List<TensorRef> Inputs { get; init; }

    /// <summary>Statements evaluated in both passes before the norm.</summary>
    public required List<string> PreStatements { get; init; }

    /// <summary>Expression whose square is accumulated.</summary>
    public required string NormExpression { get; init; }

    /// <summary>Outputs written in pass 1.</summary>
    public required List<(TensorRef Ref, string Expr)> Pass1Outputs { get; init; }

    /// <summary>Statements evaluated in pass 2 after <c>nrm</c> is known.</summary>
    public required List<string> PostStatements { get; init; }

    /// <summary>Outputs written in pass 2.</summary>
    public required List<(TensorRef Ref, string Expr)> Pass2Outputs { get; init; }

    /// <summary>True when a per-column f32 weight scales the normalized value.</summary>
    public required bool HasWeight { get; init; }

    /// <summary>0 for RMS norm (mean of squares), 1 for L2 norm (sum of squares).</summary>
    public required int Mode { get; init; }

    /// <summary>Epsilon added before the reciprocal square root.</summary>
    public required float Epsilon { get; init; }

    /// <summary>True when subgroup arithmetic may be used for the reduction.</summary>
    public required bool UseSubgroupOps { get; init; }

    /// <summary>Total number of outputs (pass 1 then pass 2).</summary>
    public int OutputCount => Pass1Outputs.Count + Pass2Outputs.Count;

    /// <summary>Builds the kernel source.</summary>
    public string Build()
    {
        _pitches = ElementwiseCodegen.Pitches(Shape);
        _rowLength = Shape[^1];
        // Fixed trip count per thread. Short rows (decode: one workgroup per row, latency bound) are fully
        // unrolled with every load issued before any store: inputs and outputs are plain pointers that may alias, so
        // an interleaved load/store loop would serialize one memory round trip per element. The loaded values stay in
        // registers for the second pass.
        _iterations = (_rowLength + Threads - 1) / Threads;
        _unroll = _iterations <= MaxUnrolledIterations;

        AppendParams();
        _sb.AppendLine($"groupshared float red[{Threads}];");
        _sb.AppendLine($"[shader(\"compute\")] [numthreads({Threads}, 1, 1)]");
        _sb.AppendLine("void main(uint3 gid : SV_GroupID, uint3 lt : SV_GroupThreadID, uniform Params p) {");
        _sb.AppendLine("  uint row = gid.x; if (row >= p.rows) return;");
        _sb.AppendLine("  float ss = 0;");
        AppendPass(second: false);
        AppendReduction();
        _sb.AppendLine(Mode == 0
            ? $"  float scale = rsqrt(tot / {GraphLowerer.F((float)_rowLength)} + {GraphLowerer.F(Epsilon)});"
            : $"  float scale = rsqrt(tot + {GraphLowerer.F(Epsilon)});");
        AppendPass(second: true);
        _sb.AppendLine("}");
        return _sb.ToString();
    }

    private void AppendParams()
    {
        _sb.AppendLine("#include \"math_ext.slang\"");
        _sb.AppendLine("struct Params {");
        for (int i = 0; i < Inputs.Count; i++)
        {
            _sb.AppendLine($"  {ElementwiseCodegen.PointerType(Inputs[i].Type)}* in{i};");
        }

        _sb.AppendLine("  float* w;");
        for (int i = 0; i < OutputCount; i++)
        {
            _sb.AppendLine($"  float* out{i};");
        }

        _sb.AppendLine("  uint rows;");
        _sb.AppendLine("};");
    }

    /// <summary>Workgroup sum of <c>ss</c> into <c>tot</c>.</summary>
    private void AppendReduction()
    {
        if (UseSubgroupOps)
        {
            // Subgroup sums, then one barrier and a short serial sum over the subgroups.
            _sb.AppendLine("  float ws = WaveActiveSum(ss);");
            _sb.AppendLine("  uint lanes = WaveGetLaneCount();");
            _sb.AppendLine("  if (WaveIsFirstLane()) red[lt.x / lanes] = ws;");
            _sb.AppendLine("  GroupMemoryBarrierWithGroupSync();");
            _sb.AppendLine($"  float tot = 0; for (uint k = 0; k < {Threads}u / lanes; k++) tot += red[k];");
            return;
        }

        _sb.AppendLine("  red[lt.x] = ss;");
        _sb.AppendLine("  GroupMemoryBarrierWithGroupSync();");
        _sb.AppendLine($"  for (uint s = {Threads / 2}u; s > 0; s >>= 1) {{ if (lt.x < s) red[lt.x] += red[lt.x + s]; GroupMemoryBarrierWithGroupSync(); }}");
        _sb.AppendLine("  float tot = red[0];");
    }

    private void AppendPass(bool second)
    {
        if (_unroll)
        {
            AppendUnrolledPass(second);
            return;
        }

        _sb.AppendLine($"  for (uint i = lt.x; i < {_rowLength}u; i += {Threads}u) {{");
        AppendIndices("    ");
        for (int i = 0; i < Inputs.Count; i++)
        {
            _sb.AppendLine($"    float a{i} = {ElementwiseCodegen.Load(Inputs[i], i, Shape)};");
        }

        AppendPassBody(second, weightExpression: "p.w[i]");
        _sb.AppendLine("  }");
    }

    private void AppendUnrolledPass(bool second)
    {
        if (!second)
        {
            // The first pass loads the row (and weights) into registers; the second pass reuses them.
            for (int i = 0; i < Inputs.Count; i++)
            {
                _sb.AppendLine($"  float ar{i}[{_iterations}];");
            }

            if (HasWeight)
            {
                _sb.AppendLine($"  float wr[{_iterations}];");
            }

            var loads = new StringBuilder();
            AppendIndices("    ", loads);
            for (int i = 0; i < Inputs.Count; i++)
            {
                loads.AppendLine($"    ar{i}[j] = {ElementwiseCodegen.Load(Inputs[i], i, Shape)};");
            }

            if (HasWeight)
            {
                loads.AppendLine("    wr[j] = p.w[i];");
            }

            AppendUnrolledLoop(loads.ToString());
        }

        var body = new StringBuilder();
        AppendIndices("    ", body);
        for (int i = 0; i < Inputs.Count; i++)
        {
            body.AppendLine($"    float a{i} = ar{i}[j];");
        }

        AppendPassBody(second, weightExpression: "wr[j]", body);
        AppendUnrolledLoop(body.ToString());
    }

    private void AppendUnrolledLoop(string body)
    {
        _sb.AppendLine($"  [ForceUnroll] for (uint j = 0; j < {_iterations}u; j++) {{");
        _sb.AppendLine($"    uint i = lt.x + j * {Threads}u;");
        bool guarded = _rowLength % Threads != 0;
        if (guarded)
        {
            _sb.AppendLine($"    if (i < {_rowLength}u) {{");
        }

        _sb.Append(body);
        if (guarded)
        {
            _sb.AppendLine("    }");
        }

        _sb.AppendLine("  }");
    }

    private void AppendIndices(string indent, StringBuilder? target = null)
    {
        var sb = target ?? _sb;
        sb.AppendLine($"{indent}uint idx = row * {_rowLength}u + i;");
        ElementwiseCodegen.AppendCoordinates(sb, indent, Shape, _pitches);
    }

    /// <summary>Pre statements, then pass 1 (outputs and sum of squares) or pass 2 (normalize, post statements, outputs).</summary>
    private void AppendPassBody(bool second, string weightExpression, StringBuilder? target = null)
    {
        var sb = target ?? _sb;
        foreach (var statement in PreStatements)
        {
            sb.AppendLine("    " + statement);
        }

        if (!second)
        {
            for (int i = 0; i < Pass1Outputs.Count; i++)
            {
                sb.AppendLine($"    p.out{i}[{ElementwiseCodegen.Offset(Pass1Outputs[i].Ref, Shape)}] = {Pass1Outputs[i].Expr};");
            }

            sb.AppendLine($"    ss += {NormExpression} * {NormExpression};");
            return;
        }

        sb.AppendLine($"    float nrm = {NormExpression} * scale{(HasWeight ? " * " + weightExpression : "")};");
        foreach (var statement in PostStatements)
        {
            sb.AppendLine("    " + statement);
        }

        for (int i = 0; i < Pass2Outputs.Count; i++)
        {
            sb.AppendLine($"    p.out{Pass1Outputs.Count + i}[{ElementwiseCodegen.Offset(Pass2Outputs[i].Ref, Shape)}] = {Pass2Outputs[i].Expr};");
        }
    }
}
