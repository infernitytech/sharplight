namespace Sharplight.Import.PyTorch;

/// <summary>A storage class global such as <c>torch.FloatStorage</c>, carrying its element type.</summary>
/// <param name="DType">The element type.</param>
internal sealed record PickleStorageType(TorchDType DType);
