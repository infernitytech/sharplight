using System.Numerics;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// numpy's <c>np.random.default_rng(seed)</c> (PCG64 seeded through <c>SeedSequence</c>) as far as checkpoints need it to rebuild
/// constants their reference code draws at load time: DeepSeek-V4.1's Engram hash multipliers come from
/// <c>default_rng(10007 * layer).integers(0, bound, size=N, dtype=np.int64)</c>. Bit-exact with numpy 1.17 and later for a non-negative
/// integer seed and bounds above 2^32 (the 64-bit Lemire path of <c>Generator.integers</c>).
/// </summary>
internal sealed class NumpyDefaultRng
{
    private const uint InitA = 0x43b0d7e5;
    private const uint MultA = 0x931e8875;
    private const uint InitB = 0x8b51f9dd;
    private const uint MultB = 0x58f38ded;
    private const uint MixMultL = 0xca01f9dd;
    private const uint MixMultR = 0x4973f715;
    private const int PoolSize = 4;
    private static readonly UInt128 s_multiplier = new(0x2360ED051FC65DA4UL, 0x4385DF649FCCF645UL);

    private readonly UInt128 _increment;
    private UInt128 _state;

    /// <summary>The generator of <c>default_rng(<paramref name="seed"/>)</c>.</summary>
    /// <param name="seed">A non-negative integer seed.</param>
    public NumpyDefaultRng(ulong seed)
    {
        var words = SeedSequenceState(seed, 4);
        var initState = new UInt128(words[0], words[1]);
        var initSequence = new UInt128(words[2], words[3]);
        _increment = (initSequence << 1) | 1;
        _state = 0;
        Step();
        _state += initState;
        Step();
    }

    /// <summary>The next value of <c>integers(0, <paramref name="high"/>, dtype=np.int64)</c>.</summary>
    /// <param name="high">Exclusive upper bound, above 2^32.</param>
    /// <exception cref="ArgumentOutOfRangeException">The bound takes numpy's 32-bit path.</exception>
    public long NextBelow(long high)
    {
        if (high <= 1L << 32)
        {
            throw new ArgumentOutOfRangeException(nameof(high), "Only bounds above 2^32 are reproduced.");
        }

        ulong range = (ulong)high;
        UInt128 m = (UInt128)Next64() * range;
        ulong leftover = (ulong)m;
        if (leftover < range)
        {
            ulong threshold = (ulong.MaxValue - (range - 1)) % range;
            while (leftover < threshold)
            {
                m = (UInt128)Next64() * range;
                leftover = (ulong)m;
            }
        }

        return (long)(ulong)(m >> 64);
    }

    private void Step() => _state = _state * s_multiplier + _increment;

    // PCG64 XSL-RR output after a step.
    private ulong Next64()
    {
        Step();
        ulong high = (ulong)(_state >> 64);
        return BitOperations.RotateRight(high ^ (ulong)_state, (int)(high >> 58));
    }

    // SeedSequence(seed).generate_state(n, np.uint64).
    private static ulong[] SeedSequenceState(ulong seed, int count)
    {
        var entropy = new List<uint>();
        do
        {
            entropy.Add((uint)seed);
            seed >>= 32;
        }
        while (seed != 0);

        uint hashConst = InitA;
        uint HashMix(uint value)
        {
            value ^= hashConst;
            hashConst *= MultA;
            value *= hashConst;
            return value ^ (value >> 16);
        }

        static uint Mix(uint x, uint y)
        {
            uint result = MixMultL * x - MixMultR * y;
            return result ^ (result >> 16);
        }

        var pool = new uint[PoolSize];
        for (int i = 0; i < PoolSize; i++)
        {
            pool[i] = HashMix(i < entropy.Count ? entropy[i] : 0);
        }

        for (int source = 0; source < PoolSize; source++)
        {
            for (int destination = 0; destination < PoolSize; destination++)
            {
                if (source != destination)
                {
                    pool[destination] = Mix(pool[destination], HashMix(pool[source]));
                }
            }
        }

        for (int source = PoolSize; source < entropy.Count; source++)
        {
            for (int destination = 0; destination < PoolSize; destination++)
            {
                pool[destination] = Mix(pool[destination], HashMix(entropy[source]));
            }
        }

        uint hash = InitB;
        var state = new ulong[count];
        for (int i = 0; i < count * 2; i++)
        {
            uint value = pool[i % PoolSize] ^ hash;
            hash *= MultB;
            value *= hash;
            value ^= value >> 16;
            state[i / 2] |= (ulong)value << (i % 2 * 32);
        }

        return state;
    }
}
