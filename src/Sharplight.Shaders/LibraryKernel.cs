using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace Sharplight.Shaders;

/// <summary>
/// Kernels a provider runs with a vendor library instead of generated code (<see cref="Runtime.DeviceFeatures.LibraryGemm"/>). The
/// lowering emits them like any kernel: their <see cref="CompiledKernel.Code"/> is a one-line descriptor
/// (<c>// sharplight-library: gemm a=f16 n=4096 k=12288</c>) that providers with the library execute, and providers without it reject
/// at load. A library GEMM's launch has the arguments <c>x</c> (dense activation rows of K values of the type <c>a</c>), <c>w</c>
/// (N rows of K weights of that type), <c>y</c> (dense f32 rows of N values) and the row count.
/// </summary>
public static class LibraryKernel
{
    /// <summary>Start of a library kernel's code.</summary>
    public const string Header = "// sharplight-library: ";

    /// <summary>A library GEMM y = x · wᵀ for <paramref name="target"/>: f32 output, <paramref name="type"/> operands, N x K weights.</summary>
    public static CompiledKernel Gemm(ShaderTarget target, string type, long n, long k)
    {
        string text = string.Create(CultureInfo.InvariantCulture, $"{Header}gemm a={type} n={n} k={k}");
        string key = "lib" + Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text + "|" + target)))[..29].ToLowerInvariant();
        return new CompiledKernel("library", "gemm", target, Encoding.UTF8.GetBytes(text), key);
    }

    /// <summary>Parses the descriptor of a library kernel's code, or returns false for generated code.</summary>
    /// <param name="code">The kernel's code.</param>
    /// <param name="operation">The operation (<c>gemm</c>).</param>
    /// <param name="parameters">Its parameters (<c>a</c>, <c>n</c>, <c>k</c>).</param>
    public static bool TryParse(ReadOnlySpan<byte> code, out string operation, out IReadOnlyDictionary<string, string> parameters)
    {
        operation = "";
        parameters = new Dictionary<string, string>();
        var header = Encoding.UTF8.GetBytes(Header);
        if (!code.StartsWith(header))
        {
            return false;
        }

        var words = Encoding.UTF8.GetString(code[header.Length..]).Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
        operation = words[0];
        parameters = words.Skip(1).Select(w => w.Split('=', 2)).ToDictionary(p => p[0], p => p[1], StringComparer.Ordinal);
        return true;
    }
}
