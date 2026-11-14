namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>Columns the weight list sorts by.</summary>
internal enum WeightColumn
{
    /// <summary>File order (offset); the default.</summary>
    Offset,

    /// <summary>Name.</summary>
    Name,

    /// <summary>Stored type.</summary>
    Type,

    /// <summary>Shape (by element count).</summary>
    Shape,

    /// <summary>Bits per element.</summary>
    Bits,

    /// <summary>Stored size.</summary>
    Size,
}
