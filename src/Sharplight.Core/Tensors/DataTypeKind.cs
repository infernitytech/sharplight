namespace Sharplight.Tensors;

/// <summary>Class of a <see cref="DataType"/>.</summary>
public enum DataTypeKind
{
    /// <summary>Floating point.</summary>
    Float,

    /// <summary>Integer.</summary>
    Int,

    /// <summary>Boolean.</summary>
    Bool,

    /// <summary>Block-quantized: several logical elements share scales and are stored in fixed-size blocks.</summary>
    Quantized,
}
