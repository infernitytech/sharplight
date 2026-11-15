using System.Buffers.Binary;
using System.Numerics;
using System.Text;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>MurmurHash3 (x86, 32 bit) of each element of a str or i32 tensor (see <c>StandardOps.MurmurHash3</c>).</summary>
internal sealed unsafe class MurmurHash3Op(TRef x, TRef y, uint seed, int elementBytes) : CpuOp
{
    public override string Name => "MurmurHash3";

    public override void Run(RunParameters p)
    {
        long n = x.Elements(p);
        Span<byte> bytes = stackalloc byte[8];
        for (long i = 0; i < n; i++)
        {
            int value = ((int*)x.P)[i];
            uint hash;
            if (x.Type == DataType.Str)
            {
                hash = Hash(Encoding.UTF8.GetBytes(p.Strings[value]), seed);
            }
            else
            {
                BinaryPrimitives.WriteInt64LittleEndian(bytes, value);
                hash = Hash(bytes[..elementBytes], seed);
            }

            ((uint*)y.P)[i] = hash;
        }
    }

    /// <summary>MurmurHash3_x86_32 of <paramref name="data"/>.</summary>
    internal static uint Hash(ReadOnlySpan<byte> data, uint seed)
    {
        const uint c1 = 0xcc9e2d51;
        const uint c2 = 0x1b873593;
        uint h = seed;
        int blocks = data.Length / 4;
        for (int b = 0; b < blocks; b++)
        {
            uint k = BinaryPrimitives.ReadUInt32LittleEndian(data[(b * 4)..]);
            k *= c1;
            k = BitOperations.RotateLeft(k, 15);
            k *= c2;
            h ^= k;
            h = BitOperations.RotateLeft(h, 13);
            h = h * 5 + 0xe6546b64;
        }

        var tail = data[(blocks * 4)..];
        uint t = 0;
        for (int j = tail.Length - 1; j >= 0; j--)
        {
            t = (t << 8) | tail[j];
        }

        if (tail.Length > 0)
        {
            t *= c1;
            t = BitOperations.RotateLeft(t, 15);
            t *= c2;
            h ^= t;
        }

        h ^= (uint)data.Length;
        h ^= h >> 16;
        h *= 0x85ebca6b;
        h ^= h >> 13;
        h *= 0xc2b2ae35;
        h ^= h >> 16;
        return h;
    }
}
