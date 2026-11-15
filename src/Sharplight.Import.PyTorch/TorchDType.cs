namespace Sharplight.Import.PyTorch;

/// <summary>Element type of a tensor stored in a PyTorch checkpoint.</summary>
public enum TorchDType
{
    /// <summary>64-bit float (<c>torch.DoubleStorage</c>).</summary>
    Float64,

    /// <summary>32-bit float (<c>torch.FloatStorage</c>).</summary>
    Float32,

    /// <summary>16-bit IEEE float (<c>torch.HalfStorage</c>).</summary>
    Float16,

    /// <summary>bfloat16 (<c>torch.BFloat16Storage</c>).</summary>
    BFloat16,

    /// <summary>64-bit signed integer (<c>torch.LongStorage</c>).</summary>
    Int64,

    /// <summary>32-bit signed integer (<c>torch.IntStorage</c>).</summary>
    Int32,

    /// <summary>16-bit signed integer (<c>torch.ShortStorage</c>).</summary>
    Int16,

    /// <summary>8-bit signed integer (<c>torch.CharStorage</c>).</summary>
    Int8,

    /// <summary>8-bit unsigned integer (<c>torch.ByteStorage</c>).</summary>
    UInt8,

    /// <summary>Boolean, one byte per element (<c>torch.BoolStorage</c>).</summary>
    Bool,
}

/// <summary>Helpers for <see cref="TorchDType"/>.</summary>
public static class TorchDTypeExtensions
{
    extension(TorchDType dtype)
    {
        /// <summary>Size of one element in bytes.</summary>
        public int ElementSize => dtype switch
        {
            TorchDType.Float64 or TorchDType.Int64 => 8,
            TorchDType.Float32 or TorchDType.Int32 => 4,
            TorchDType.Float16 or TorchDType.BFloat16 or TorchDType.Int16 => 2,
            _ => 1,
        };
    }
}
