namespace Sharplight.Cpu;

/// <summary>
/// A float vector of a fixed width, so one kernel can be written once and instantiated for 256-bit and 512-bit vectors
/// (the JIT specializes generic code over these structs and inlines every member).
/// </summary>
internal unsafe interface IFloatLanes<TSelf>
    where TSelf : struct, IFloatLanes<TSelf>
{
    /// <summary>Number of float lanes.</summary>
    public static abstract int Count { get; }

    /// <summary>All lanes zero.</summary>
    public static abstract TSelf Zero { get; }

    /// <summary>All lanes <paramref name="value"/>.</summary>
    public static abstract TSelf Broadcast(float value);

    /// <summary>Loads <see cref="Count"/> floats.</summary>
    public static abstract TSelf Load(float* p);

    /// <summary>Stores <see cref="Count"/> floats.</summary>
    public static abstract void Store(TSelf v, float* p);

    /// <summary>a * b + c.</summary>
    public static abstract TSelf Fma(TSelf a, TSelf b, TSelf c);

    /// <summary>a * b.</summary>
    public static abstract TSelf Multiply(TSelf a, TSelf b);

    /// <summary>a + b.</summary>
    public static abstract TSelf Add(TSelf a, TSelf b);

    /// <summary>a - b.</summary>
    public static abstract TSelf Subtract(TSelf a, TSelf b);

    /// <summary>Lane-wise maximum.</summary>
    public static abstract TSelf Max(TSelf a, TSelf b);

    /// <summary>Lane-wise e^x.</summary>
    public static abstract TSelf Exp(TSelf x);

    /// <summary>Lanes equal to negative infinity become zero.</summary>
    public static abstract TSelf ZeroIfNegativeInfinity(TSelf v);

    /// <summary>Keeps lane r of <paramref name="v"/> when <paramref name="first"/>[r] &lt;= <paramref name="j"/> &lt; <paramref name="end"/>[r], else negative infinity.</summary>
    public static abstract TSelf KeepInRange(TSelf v, int j, int* first, int* end);
}
