using System.Buffers.Binary;
using Sharplight.Shaders.Lowering;

namespace Sharplight.Cuda;

/// <summary>
/// What recording a launch with one of its variants resolves independently of the symbol bindings: the loaded kernel, the profile name
/// and the parameter struct with every pointer, constant scalar and tensor map filled in (the segment's buffers do not move). A new step
/// shape then only evaluates the grid and writes the symbol-valued scalars into a copy of it, instead of looking up the kernel, the
/// buffers and the tensor maps of every launch again.
/// </summary>
/// <param name="kernel">The loaded kernel of the variant.</param>
/// <param name="profileName">The launch's name as profiles show it.</param>
/// <param name="packed">The parameter struct; its symbol-valued scalars are written per binding (<paramref name="symbols"/>).</param>
/// <param name="symbols">Byte offset, symbol and multiplier of each scalar that depends on the bindings.</param>
internal sealed class CudaLaunchTemplate(CudaKernel kernel, string profileName, byte[] packed, (int Offset, string Symbol, long Multiplier)[] symbols)
{
    /// <summary>The loaded kernel of the variant.</summary>
    public CudaKernel Kernel => kernel;

    /// <summary>The launch's name as profiles show it.</summary>
    public string ProfileName => profileName;

    /// <summary>
    /// The parameter struct for <paramref name="bindings"/>: the template itself when no scalar depends on the bindings (recordings only
    /// read it), else a copy with those scalars written.
    /// </summary>
    public byte[] Arguments(Bindings bindings)
    {
        if (symbols.Length == 0)
        {
            return packed;
        }

        var arguments = (byte[])packed.Clone();
        foreach (var (offset, symbol, multiplier) in symbols)
        {
            BinaryPrimitives.WriteUInt32LittleEndian(arguments.AsSpan(offset), (uint)(bindings[symbol] * multiplier));
        }

        return arguments;
    }
}
