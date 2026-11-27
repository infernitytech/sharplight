namespace Sharplight.Import.HuggingFace;

/// <summary>
/// numpy's <c>np.random.default_rng(seed).integers(0, high, size=n)</c> (int64): a <c>SeedSequence</c> of the seed feeds the PCG64 generator
/// (128-bit LCG, XSL-RR output) and the bounded draws use Lemire's method (64-bit draws above 2³² − 1, buffered 32-bit halves below), as
/// numpy &gt;= 1.17 does. Checkpoints that derive constants from numpy's generator (Hunyuan v4 Flash's Engram hash multipliers) need the same
/// values.
/// </summary>
internal static class NumpyRandom
{
    private const uint InitA = 0x43b0d7e5;
    private const uint MultA = 0x931e8875;
    private const uint InitB = 0x8b51f9dd;
    private const uint MultB = 0x58f38ded;
    private const uint MixMultL = 0xca01f9dd;
    private const uint MixMultR = 0x4973f715;
    private const int XShift = 16;
    private const int PoolSize = 4;

    private static readonly UInt128 s_multiplier = new(2549297995355413924UL, 4865540595714422341UL);

    /// <summary>The draws of <c>default_rng(seed).integers(0, high, size=count)</c>.</summary>
    /// <param name="seed">A non-negative seed.</param>
    /// <param name="high">Exclusive upper bound, at least 1.</param>
    /// <param name="count">Number of draws.</param>
    /// <exception cref="ArgumentOutOfRangeException">The seed is negative or the bound below 1.</exception>
    public static long[] Integers(long seed, long high, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(seed);
        ArgumentOutOfRangeException.ThrowIfLessThan(high, 1L);
        var generator = new Pcg64(GenerateState(Pool(seed)));
        ulong range = (ulong)high - 1;
        var values = new long[count];
        ulong buffer = 0;
        int buffered = 0;
        for (int i = 0; i < count; i++)
        {
            values[i] = range == 0 ? 0 : range <= uint.MaxValue ? (long)BoundedUInt32(ref generator, (uint)range, ref buffer, ref buffered)
                : (long)BoundedUInt64(ref generator, range);
        }

        return values;
    }

    // Lemire's bounded draw on 64-bit outputs (numpy bounded_lemire_uint64).
    private static ulong BoundedUInt64(ref Pcg64 generator, ulong range)
    {
        ulong exclusive = range + 1;
        UInt128 m = (UInt128)generator.Next() * exclusive;
        ulong leftover = (ulong)m;
        if (leftover < exclusive)
        {
            ulong threshold = (ulong.MaxValue - range) % exclusive;
            while (leftover < threshold)
            {
                m = (UInt128)generator.Next() * exclusive;
                leftover = (ulong)m;
            }
        }

        return (ulong)(m >> 64);
    }

    // Lemire's bounded draw on 32-bit halves of the 64-bit outputs, low half first (numpy buffered_bounded_lemire_uint32).
    private static uint BoundedUInt32(ref Pcg64 generator, uint range, ref ulong buffer, ref int buffered)
    {
        if (range == uint.MaxValue)
        {
            return Next32(ref generator, ref buffer, ref buffered);
        }

        uint exclusive = range + 1;
        ulong m = (ulong)Next32(ref generator, ref buffer, ref buffered) * exclusive;
        uint leftover = (uint)m;
        if (leftover < exclusive)
        {
            uint threshold = (uint.MaxValue - range) % exclusive;
            while (leftover < threshold)
            {
                m = (ulong)Next32(ref generator, ref buffer, ref buffered) * exclusive;
                leftover = (uint)m;
            }
        }

        return (uint)(m >> 32);
    }

    private static uint Next32(ref Pcg64 generator, ref ulong buffer, ref int buffered)
    {
        if (buffered == 0)
        {
            buffer = generator.Next();
            buffered = 1;
        }
        else
        {
            buffer >>= 32;
            buffered = 0;
        }

        return (uint)buffer;
    }

    // SeedSequence(seed).pool: the entropy's 32-bit words mixed into four.
    private static uint[] Pool(long seed)
    {
        var entropy = new List<uint>();
        ulong n = (ulong)seed;
        do
        {
            entropy.Add((uint)n);
            n >>= 32;
        }
        while (n > 0);

        var mixer = new uint[PoolSize];
        uint hashConst = InitA;
        for (int i = 0; i < PoolSize; i++)
        {
            mixer[i] = HashMix(i < entropy.Count ? entropy[i] : 0u, ref hashConst);
        }

        for (int source = 0; source < PoolSize; source++)
        {
            for (int target = 0; target < PoolSize; target++)
            {
                if (source != target)
                {
                    mixer[target] = Mix(mixer[target], HashMix(mixer[source], ref hashConst));
                }
            }
        }

        for (int source = PoolSize; source < entropy.Count; source++)
        {
            for (int target = 0; target < PoolSize; target++)
            {
                mixer[target] = Mix(mixer[target], HashMix(entropy[source], ref hashConst));
            }
        }

        return mixer;
    }

    // SeedSequence.generate_state(4, uint64): eight 32-bit words, paired low word first.
    private static ulong[] GenerateState(uint[] pool)
    {
        var words = new uint[8];
        uint hashConst = InitB;
        for (int i = 0; i < words.Length; i++)
        {
            uint value = pool[i % PoolSize] ^ hashConst;
            hashConst *= MultB;
            value *= hashConst;
            value ^= value >> XShift;
            words[i] = value;
        }

        return [.. Enumerable.Range(0, 4).Select(k => words[2 * k] | ((ulong)words[2 * k + 1] << 32))];
    }

    private static uint HashMix(uint value, ref uint hashConst)
    {
        value ^= hashConst;
        hashConst *= MultA;
        value *= hashConst;
        value ^= value >> XShift;
        return value;
    }

    private static uint Mix(uint x, uint y)
    {
        uint result = MixMultL * x - MixMultR * y;
        result ^= result >> XShift;
        return result;
    }

    /// <summary>numpy's PCG64 (pcg_setseq_128_xsl_rr_64) seeded with state words [seed_hi, seed_lo, inc_hi, inc_lo].</summary>
    private struct Pcg64
    {
        private UInt128 _state;
        private readonly UInt128 _increment;

        public Pcg64(ulong[] seed)
        {
            _state = 0;
            _increment = (new UInt128(seed[2], seed[3]) << 1) | 1;
            Step();
            _state += new UInt128(seed[0], seed[1]);
            Step();
        }

        public ulong Next()
        {
            Step();
            ulong high = (ulong)(_state >> 64);
            ulong low = (ulong)_state;
            return ulong.RotateRight(high ^ low, (int)(_state >> 122));
        }

        private void Step() => _state = _state * s_multiplier + _increment;
    }
}
