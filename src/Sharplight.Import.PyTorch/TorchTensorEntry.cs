namespace Sharplight.Import.PyTorch;

/// <summary>A tensor of a PyTorch checkpoint: where its contiguous bytes lie inside the file.</summary>
/// <param name="Name">The state-dict key (nested dictionaries are joined with '.').</param>
/// <param name="DType">The element type.</param>
/// <param name="Shape">Dimensions, outermost first.</param>
/// <param name="FileOffset">Absolute offset of the first element in the checkpoint file.</param>
/// <param name="ByteLength">Size of the tensor data in bytes.</param>
public sealed record TorchTensorEntry(string Name, TorchDType DType, long[] Shape, long FileOffset, long ByteLength);
