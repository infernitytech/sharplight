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
    private CudaPtx(byte[] code, string function, uint x, uint y, uint z)
    {
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
            uint.Parse(threads.Groups[3].Value, CultureInfo.InvariantCulture));
    }

    [GeneratedRegex(@"^// sharplight-threads: (\d+) (\d+) (\d+)$", RegexOptions.Multiline)]
    private static partial Regex ThreadsPattern();

    [GeneratedRegex(@"\.entry\s+(\w+)\s*\(")]
    private static partial Regex EntryPattern();
}
