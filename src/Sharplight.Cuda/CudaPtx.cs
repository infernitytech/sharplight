using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;
using Sharplight.Shaders;

namespace Sharplight.Cuda;

/// <summary>
/// The PTX of one compiled kernel with what loading it needs: the name of its <c>.entry</c> (Slang renames entry points) and the block
/// size, which PTX does not carry (kept in a header line, see <see cref="SlangCompiler.CudaThreadsHeader"/>).
/// </summary>
internal sealed partial class CudaPtx
{
    /// <summary>Start of the header line after the block size of a kernel that uses dynamic shared memory, followed by its bytes.</summary>
    public const string SharedHeader = "// sharplight-shared: ";

    /// <summary>
    /// Header line of a kernel whose entry starts with <c>griddepcontrol.wait</c> and <c>griddepcontrol.launch_dependents</c> (compute
    /// capability 9.0 and later): it waits for the previous launch's completion and memory itself, so it may be launched as a
    /// programmatic dependent of it (<see cref="CudaKernel.WaitsForPrevious"/>).
    /// </summary>
    public const string DependentHeader = "// sharplight-dependent";

    private CudaPtx(byte[] code, string function, uint x, uint y, uint z, int sharedBytes, CudaTensorMapSpec[] tensorMaps, bool waitsForPrevious)
    {
        WaitsForPrevious = waitsForPrevious;
        SharedBytes = sharedBytes;
        TensorMaps = tensorMaps;
        Code = code;
        Function = function;
        ThreadsX = x;
        ThreadsY = y;
        ThreadsZ = z;
    }

    /// <summary>The PTX text (with the header line), NUL-terminated for the driver.</summary>
    public byte[] Code { get; }

    public string Function { get; }

    public uint ThreadsX { get; }

    public uint ThreadsY { get; }

    public uint ThreadsZ { get; }

    /// <summary>Dynamic shared memory the kernel needs at launch (bytes), 0 for kernels with only static shared memory.</summary>
    public int SharedBytes { get; }

    /// <summary>Tensor maps at the end of the kernel's parameters (empty for most kernels).</summary>
    public CudaTensorMapSpec[] TensorMaps { get; }

    /// <summary>True when the kernel waits for the previous launch itself (<see cref="DependentHeader"/>).</summary>
    public bool WaitsForPrevious { get; }

    /// <summary>Parses the code of a kernel compiled for <see cref="ShaderTarget.Ptx"/> by the <see cref="NvrtcTranslator"/>.</summary>
    /// <param name="code">The kernel's code.</param>
    /// <param name="what">The kernel, for messages.</param>
    /// <exception cref="InvalidOperationException">The code has no threads header or no entry.</exception>
    public static CudaPtx Parse(byte[] code, string what)
    {
        string text = Encoding.UTF8.GetString(code);
        var threads = ThreadsPattern().Match(text);
        var entry = EntryPattern().Match(text);
        if (!threads.Success || !entry.Success)
        {
            throw new InvalidOperationException($"Kernel {what} is not PTX with a block size.");
        }

        return new CudaPtx([.. code, 0], entry.Groups[1].Value,
            uint.Parse(threads.Groups[1].Value, CultureInfo.InvariantCulture),
            uint.Parse(threads.Groups[2].Value, CultureInfo.InvariantCulture),
            uint.Parse(threads.Groups[3].Value, CultureInfo.InvariantCulture),
            SharedPattern().Match(text) is { Success: true } shared ? int.Parse(shared.Groups[1].Value, CultureInfo.InvariantCulture) : 0,
            TensorMapsPattern().Match(text) is { Success: true } maps ? CudaTensorMapSpec.ParseList(maps.Groups[1].Value) : [],
            DependentPattern().IsMatch(text));
    }

    [GeneratedRegex(@"^// sharplight-threads: (\d+) (\d+) (\d+)$", RegexOptions.Multiline)]
    private static partial Regex ThreadsPattern();

    [GeneratedRegex(@"^// sharplight-shared: (\d+)$", RegexOptions.Multiline)]
    private static partial Regex SharedPattern();

    [GeneratedRegex(@"^// sharplight-tensormaps: (.+)$", RegexOptions.Multiline)]
    private static partial Regex TensorMapsPattern();

    [GeneratedRegex(@"^// sharplight-dependent$", RegexOptions.Multiline)]
    private static partial Regex DependentPattern();

    [GeneratedRegex(@"\.entry\s+(\w+)\s*\(")]
    private static partial Regex EntryPattern();
}
