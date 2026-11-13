using Sharplight.Runtime;

namespace Sharplight.Cpu;

/// <summary>
/// Coordinates of the non-zero elements of a dense input as i32 rows <c>[count, rank]</c>; publishes the count as the dynamic symbol
/// <paramref name="countSymbol"/>.
/// </summary>
internal sealed unsafe class NonZeroOp(TRef x, TRef y, string countSymbol) : CpuOp
{
    public override string Name => "NonZero";

    public override void Run(RunParameters p)
    {
        var dims = OpHelpers.Dims(x, p);
        long total = OpHelpers.Prod(dims, 0, dims.Length);
        int rank = dims.Length;
        var coords = new long[rank];
        int* output = (int*)y.P;
        long count = 0;
        for (long i = 0; i < total; i++)
        {
            if (OpHelpers.IsNonZero(x, i))
            {
                for (int k = 0; k < rank; k++)
                {
                    output[count * rank + k] = (int)coords[k];
                }

                count++;
            }

            for (int k = rank - 1; k >= 0; k--)
            {
                if (++coords[k] < dims[k])
                {
                    break;
                }

                coords[k] = 0;
            }
        }

        p.Symbols[countSymbol] = count;
    }
}
