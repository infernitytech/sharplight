using System.Runtime.CompilerServices;

namespace Sharplight.Tokenizers.Bpe;

/// <summary>Open-addressing hash from a (left symbol, right symbol) pair to (rank, merged symbol).</summary>
internal sealed class MergeTable
{
    private readonly long[] _keys;
    private readonly int[] _ranks;
    private readonly int[] _results;
    private readonly int _mask;
    private readonly int _shift;

    public MergeTable(int expected)
    {
        int cap = 16;
        while (cap < expected * 2) cap <<= 1;
        _keys = new long[cap];
        Array.Fill(_keys, -1L);
        _ranks = new int[cap];
        _results = new int[cap];
        _mask = cap - 1;
        _shift = 64 - System.Numerics.BitOperations.Log2((uint)cap);
    }

    public int Count { get; private set; }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int Slot(long key) => (int)(((ulong)key * 0x9E3779B97F4A7C15UL) >> _shift);

    /// <summary>Adds a pair; the first (lowest rank) registration wins.</summary>
    public bool TryAdd(int left, int right, int rank, int result)
    {
        long key = ((long)left << 32) | (uint)right;
        int i = Slot(key);
        while (true)
        {
            long k = _keys[i];
            if (k == key) return false;
            if (k == -1)
            {
                _keys[i] = key; _ranks[i] = rank; _results[i] = result; Count++;
                return true;
            }
            i = (i + 1) & _mask;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGet(int left, int right, out int rank, out int result)
    {
        long key = ((long)left << 32) | (uint)right;
        int i = Slot(key);
        while (true)
        {
            long k = _keys[i];
            if (k == key) { rank = _ranks[i]; result = _results[i]; return true; }
            if (k == -1) { rank = 0; result = 0; return false; }
            i = (i + 1) & _mask;
        }
    }
}
