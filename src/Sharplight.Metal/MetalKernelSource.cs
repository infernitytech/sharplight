using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Shaders;

namespace Sharplight.Metal;

/// <summary>
/// The Metal Shading Language source of one compiled kernel, with what pipeline creation needs from it: the name of the
/// <c>[[kernel]]</c> function (Slang renames entry points) and the threadgroup size, which MSL does not carry (Slang records the
/// entry point's <c>[numthreads]</c> in a header line, see <see cref="SlangCompiler.MetalThreadsHeader"/>).
/// </summary>
internal sealed partial class MetalKernelSource
{
    private MetalKernelSource(string source, string function, uint x, uint y, uint z)
    {
        Source = source;
        Function = function;
        ThreadsX = x;
        ThreadsY = y;
        ThreadsZ = z;
    }

    public string Source { get; }

    public string Function { get; }

    public uint ThreadsX { get; }

    public uint ThreadsY { get; }

    public uint ThreadsZ { get; }

    public uint Threads => ThreadsX * ThreadsY * ThreadsZ;

    /// <summary>Parses the code of a kernel compiled for <see cref="ShaderTarget.Metal"/>.</summary>
    /// <exception cref="InvalidOperationException">The code has no threads header or no kernel function.</exception>
    public static MetalKernelSource Parse(CompiledKernel kernel)
    {
        string source = Encoding.UTF8.GetString(kernel.Code);
        var threads = ThreadsPattern().Match(source);
        var function = KernelPattern().Match(source);
        if (!threads.Success || !function.Success)
        {
            throw new InvalidOperationException($"Kernel {kernel.Name}:{kernel.EntryPoint} is not Metal code with a threadgroup size.");
        }

        return new MetalKernelSource(source, function.Groups[1].Value,
            uint.Parse(threads.Groups[1].Value, CultureInfo.InvariantCulture),
            uint.Parse(threads.Groups[2].Value, CultureInfo.InvariantCulture),
            uint.Parse(threads.Groups[3].Value, CultureInfo.InvariantCulture));
    }

    [GeneratedRegex(@"^// sharplight-threads: (\d+) (\d+) (\d+)$", RegexOptions.Multiline)]
    private static partial Regex ThreadsPattern();

    [GeneratedRegex(@"\[\[kernel\]\]\s+void\s+(\w+)\s*\(")]
    private static partial Regex KernelPattern();
}
