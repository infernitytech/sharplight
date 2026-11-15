namespace Sharplight.Import.PyTorch;

/// <summary>A persistent-id storage reference: which stored blob, how many elements it has, and where a legacy view starts.</summary>
/// <param name="Key">The storage key (the name of the data entry, or the legacy storage key).</param>
/// <param name="DType">The element type.</param>
/// <param name="ElementCount">Number of elements in the storage.</param>
/// <param name="ElementOffset">Offset in elements into the stored blob (nonzero for legacy storage views).</param>
internal sealed record PickleStorageRef(string Key, TorchDType DType, long ElementCount, long ElementOffset);
