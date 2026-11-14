namespace Sharplight.Tensors;

/// <summary>How a value is rounded to a power-of-two format such as <see cref="NumericFormat.F8E8M0"/>.</summary>
public enum ExponentRounding
{
    /// <summary>Round up to the next power of two unless the value already is one.</summary>
    Up,

    /// <summary>Truncate to the power of two at or below the value.</summary>
    Down,

    /// <summary>Round to the nearest power of two (ties up).</summary>
    Nearest,
}
