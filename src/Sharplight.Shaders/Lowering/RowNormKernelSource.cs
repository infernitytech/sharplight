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

    /// <summary>
    /// Pass-2 outputs (by index) written only while the step's row count <c>p.rows</c> satisfies the given Slang condition;
    /// others are always written. f16 outputs store the value rounded as <c>f32tof16</c> does.
    /// </summary>
    public IReadOnlyDictionary<int, string> Pass2Guards { get; init; } = new Dictionary<int, string>();

    /// <summary>
    /// A pass-2 output (by index) that receives the value as E4M3 activation rows of the fp8 GEMM (per row: the codes, then one
    /// f32 scale per 128 values, as <c>quantize_fp8</c> writes them) for steps of <c>MinRows</c> rows or more. The row's values
    /// pass through shared memory, so that each subgroup quantizes 128 consecutive ones; the arithmetic is quantize_fp8's.
    /// Requires 32-lane subgroups and a row length that is a multiple of 128.
    /// </summary>
    public (int Index, long MinRows)? Fp8Output { get; init; }

    /// <summary>
    /// Row capacity of the dense copy of the <see cref="Fp8Output"/> scales (<c>[row length / 128][capacity]</c> f32 after that many
    /// rows, read by the hand-written FP8 GEMM), or 0 for none.
    /// </summary>
    public long Fp8DenseScaleCapacity { get; init; }

    /// <summary>
    /// A pass-2 output (by index) that receives the value as the q8_1 activation rows of the integer-dot matvecs (per reader row of
    /// <c>ReaderRow</c> values, a multiple of the row length: the int8 codes, then per 32 values the f32 pair (d, d * sum of codes), as
    /// <c>quantize_q8_1</c> writes them) while the Slang condition <c>Guard</c> on <c>p.rows</c> holds. The row's values pass through
    /// shared memory, so that eight lanes quantize 32 consecutive ones; the arithmetic is quantize_q8_1's, so the rows are the same.
    /// Requires subgroup operations and a row length that is a multiple of 32. Not with <see cref="Fp8Output"/>.
    /// </summary>
    public (int Index, long ReaderRow, string Guard)? Q8Output { get; init; }

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
        if (Fp8Output is not null || Q8Output is not null)
        {
            _sb.AppendLine($"groupshared float qrow[{_rowLength}];");
        }

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
        AppendFp8Quantization();
        AppendQ8Quantization();
        _sb.AppendLine("}");
        return _sb.ToString();
    }

    /// <summary>The q8_1 rows of <see cref="Q8Output"/> from the values pass 2 left in <c>qrow</c>, as <c>quantize_q8_1</c> computes them.</summary>
    private void AppendQ8Quantization()
    {
        if (Q8Output is not { } q8)
        {
            return;
        }

        long quads = _rowLength / 4;
        long readerWords = (q8.ReaderRow + q8.ReaderRow / 4) / 4;
        long rowsPerReader = q8.ReaderRow / _rowLength;
        int output = Pass1Outputs.Count + q8.Index;
        _sb.AppendLine($"  if ({q8.Guard}) {{");
        _sb.AppendLine("    GroupMemoryBarrierWithGroupSync();");
        _sb.AppendLine($"    uint* qr = (uint*)p.out{output} + (row / {rowsPerReader}u) * {readerWords}u;");
        _sb.AppendLine($"    uint part = row % {rowsPerReader}u;");
        _sb.AppendLine("    uint lane = WaveGetLaneIndex();");
        _sb.AppendLine($"    for (uint c0 = 0; c0 < {quads}u; c0 += {Threads}u) {{");
        _sb.AppendLine("      uint c = c0 + lt.x;");
        _sb.AppendLine($"      bool live = c < {quads}u;");
        _sb.AppendLine("      float4 v = live ? float4(qrow[c * 4], qrow[c * 4 + 1], qrow[c * 4 + 2], qrow[c * 4 + 3]) : float4(0.0, 0.0, 0.0, 0.0);");
        _sb.AppendLine("      float amax = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));");
        _sb.AppendLine("      [ForceUnroll] for (uint m = 1; m < 8; m <<= 1) amax = max(amax, WaveReadLaneAt(amax, lane ^ m));");
        _sb.AppendLine("      float d = amax / 127.0;");
        _sb.AppendLine("      float id = d > 0 ? 1.0 / d : 0.0;");
        _sb.AppendLine("      int4 q = int4(round(v * id));");
        _sb.AppendLine("      int s = q.x + q.y + q.z + q.w;");
        _sb.AppendLine("      [ForceUnroll] for (uint m = 1; m < 8; m <<= 1) s += WaveReadLaneAt(s, lane ^ m);");
        _sb.AppendLine("      if (live) {");
        _sb.AppendLine($"        qr[part * {quads}u + c] = (uint(q.x) & 0xFF) | ((uint(q.y) & 0xFF) << 8) | ((uint(q.z) & 0xFF) << 16) | (uint(q.w) << 24);");
        _sb.AppendLine($"        if ((c & 7u) == 0) {{ uint b = {q8.ReaderRow / 4}u + (part * {_rowLength / 32}u + c / 8u) * 2u; qr[b] = asuint(d); qr[b + 1] = asuint(d * float(s)); }}");
        _sb.AppendLine("      }");
        _sb.AppendLine("    }");
        _sb.AppendLine("  }");
    }

    /// <summary>The E4M3 rows of <see cref="Fp8Output"/> from the values pass 2 left in <c>qrow</c>.</summary>
    private void AppendFp8Quantization()
    {
        if (Fp8Output is not { } fp8)
        {
            return;
        }

        long words = _rowLength / 4 + _rowLength / 128;
        int output = Pass1Outputs.Count + fp8.Index;
        _sb.AppendLine($"  if (p.rows >= {fp8.MinRows}u) {{");
        _sb.AppendLine("    GroupMemoryBarrierWithGroupSync();");
        _sb.AppendLine($"    uint* qr = (uint*)p.out{output} + row * {words}u;");
        _sb.AppendLine($"    for (uint c = lt.x; c < {_rowLength / 4}u; c += {Threads}u) {{");
        _sb.AppendLine("      float4 v = float4(qrow[c * 4], qrow[c * 4 + 1], qrow[c * 4 + 2], qrow[c * 4 + 3]);");
        _sb.AppendLine("      float d = e4m3_group_scale(v);");
        _sb.AppendLine("      qr[c] = e4m3_pack4(v, d);");
        _sb.AppendLine($"      if ((c & 31u) == 0) qr[{_rowLength / 4}u + c / 32u] = asuint(d);");
        if (Fp8DenseScaleCapacity > 0)
        {
            long dense = Fp8DenseScaleCapacity * words;
            _sb.AppendLine($"      if ((c & 31u) == 0) ((uint*)p.out{output})[{dense}u + (c / 32u) * {Fp8DenseScaleCapacity}u + row] = asuint(d);");
        }
        _sb.AppendLine("    }");
        _sb.AppendLine("  }");
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
        foreach (var (output, i) in Pass1Outputs.Concat(Pass2Outputs).Select((o, i) => (o.Ref, i)))
        {
            _sb.AppendLine($"  {ElementwiseCodegen.PointerType(output.Type)}* out{i};");
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
            var (output, expression) = Pass2Outputs[i];
            if (Fp8Output is { } fp8 && fp8.Index == i)
            {
                sb.AppendLine($"    if (p.rows >= {fp8.MinRows}u) qrow[i] = {expression};");
                continue;
            }

            if (Q8Output is { } q8 && q8.Index == i)
            {
                sb.AppendLine($"    if ({q8.Guard}) qrow[i] = {expression};");
                continue;
            }

            string guard = Pass2Guards.TryGetValue(i, out var condition) ? $"if ({condition}) " : "";
            sb.AppendLine($"    {guard}p.out{Pass1Outputs.Count + i}[{ElementwiseCodegen.Offset(output, Shape)}] = {ElementwiseCodegen.Store(output.Type, expression)};");
        }
    }
}
