namespace Sharplight.Import.PyTorch;

/// <summary>A tensor as the pickle describes it: a storage plus offset, shape and strides. No data is read.</summary>
/// <param name="Storage">The storage.</param>
/// <param name="StorageOffset">Offset of the first element into the storage, in elements.</param>
/// <param name="Shape">Dimensions.</param>
/// <param name="Strides">Strides in elements.</param>
internal sealed record PickleTensorRef(PickleStorageRef Storage, long StorageOffset, long[] Shape, long[] Strides);
