using System.Globalization;
using System.Text;

namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Writes the generated epilogue of an expert-grouped GEMM in one of two forms: Slang, included before the kernel library file
/// (<c>MOE_EPI</c> / <c>MOE_EPI1</c> over <c>MOE_EPI_FIELDS</c>, see matmul.slang and mmvq.slang), or CUDA C++ for the hand-written
/// <c>gemm_moe_native.cu</c> (<c>NM_EPI2</c> / <c>NM_EPI1</c> over <c>NM_EPI_FIELDS</c>), whose expressions are the Slang ones with
/// single-precision literals and <c>__expf</c> (as the CUDA form of the Slang kernels), so both round alike.
/// </summary>
internal sealed class MoeEpilogueWriter(StringBuilder text, bool cuda)
{
    /// <summary>The lines before the epilogue's function: the Slang math library, or the CUDA helpers the expressions may call.</summary>
    public void Header()
    {
        if (!cuda)
        {
            text.AppendLine("#include \"math_ext.slang\"");
            return;
        }

        text.AppendLine("static __device__ __forceinline__ float gelu_tanh(float x) { return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }");
        text.AppendLine("static __device__ __forceinline__ float gelu_erf(float x) { return 0.5f * x * (1.0f + erff(x * 0.7071067811865476f)); }");
    }

    /// <summary>The kernel parameters the epilogue reads (pointer declarations).</summary>
    public void Fields(string fields)
        => text.AppendLine(CultureInfo.InvariantCulture, $"#define {(cuda ? "NM_EPI_FIELDS" : "MOE_EPI_FIELDS")} {fields}");

    /// <summary>Opens the epilogue's function <paramref name="name"/> with the parameter list <paramref name="parameters"/>.</summary>
    public void Function(string name, string parameters)
        => text.AppendLine(CultureInfo.InvariantCulture, $"{(cuda ? "static __device__ __forceinline__ " : "")}float {name}({parameters}) {{");

    /// <summary>A local of the function: <c>float <paramref name="local"/> = <paramref name="expression"/>;</c>.</summary>
    public void Line(string local, string expression)
        => text.AppendLine(CultureInfo.InvariantCulture, $"  float {local} = {(cuda ? GraphLowerer.CudaExpression(expression) : expression)};");

    /// <summary>Closes the function, returning <paramref name="value"/>.</summary>
    public void Return(string value)
    {
        text.AppendLine(CultureInfo.InvariantCulture, $"  return {value};");
        text.AppendLine("}");
    }

    /// <summary>The dual epilogue macro over the products m0 and m1 of expert e, output column n.</summary>
    public void Dual(string call)
        => text.AppendLine(CultureInfo.InvariantCulture, $"#define {(cuda ? "NM_EPI2" : "MOE_EPI")}(m0, m1, e, n) {Literals(call)}");

    /// <summary>The single-product epilogue macro over the product m0 of expert e, output column n, token t and slot s.</summary>
    public void Single(string call)
        => text.AppendLine(CultureInfo.InvariantCulture, $"#define {(cuda ? "NM_EPI1" : "MOE_EPI1")}(m0, e, n, t, s) {Literals(call)}");

    /// <summary>The 64-bit index of <paramref name="value"/> in the epilogue's table subscripts.</summary>
    public string U64(string value) => cuda ? $"(unsigned long long)({value})" : $"uint64_t({value})";

    /// <summary>The kernel library file included after the epilogue (Slang only: the CUDA source precedes its file).</summary>
    public void Footer(string library)
    {
        if (!cuda)
        {
            text.AppendLine(CultureInfo.InvariantCulture, $"#include \"{library}\"");
        }
    }

    private string Literals(string expression) => cuda ? GraphLowerer.CudaExpression(expression) : expression;
}
