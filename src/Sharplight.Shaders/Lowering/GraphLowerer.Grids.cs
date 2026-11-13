using Sharplight.Tensors;

namespace Sharplight.Shaders.Lowering;

public sealed partial class GraphLowerer
{
    /// <summary>Constant buffers holding the IQ lookup grids (see <see cref="GridTables"/>), by grid key.</summary>
    private readonly Dictionary<string, BufferDecl> _gridBuffers = [];

    /// <summary>
    /// Kernels decoding an IQ weight format read its lookup grid through a pointer that is the first member of their parameter block
    /// (<c>HAS_GRID</c> in common.slang). This prepends that pointer to the arguments of every launch that passes such a weight.
    /// </summary>
    private IReadOnlyList<KernelArg> WithGridArgument(IReadOnlyList<KernelArg> args)
    {
        foreach (var arg in args)
        {
            if (arg is PointerArg { Buffer.Weight: { } weight } && GridTables.KeyOf(weight.Type) is { } key)
            {
                if (!_gridBuffers.TryGetValue(key, out var buffer))
                {
                    byte[] content = GridTables.Build(key);
                    buffer = NewBuffer(BufferKind.Weight, "grid:" + key, content.Length, init: content);
                    _gridBuffers[key] = buffer;
                }

                return [new PointerArg(buffer, 0), .. args];
            }
        }

        return args;
    }

    /// <summary>Whether a weight of this type is decoded through a lookup grid (and so needs the grid argument).</summary>
    private static bool UsesGrid(DataType type) => GridTables.KeyOf(type) is not null;
}
