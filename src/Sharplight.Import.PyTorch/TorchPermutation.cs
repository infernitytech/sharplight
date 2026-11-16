namespace Sharplight.Import.PyTorch;

/// <summary>Reorders tensors saved as a permuted view (<see cref="TorchTensorEntry.Permutation"/>) into their row-major logical order.</summary>
public static class TorchPermutation
{
    /// <summary>
    /// The elements of a tensor in storage order (shape <paramref name="storageShape"/>) as the row-major logical tensor whose axis <c>i</c> is storage
    /// axis <c>permutation[i]</c>.
    /// </summary>
    /// <typeparam name="T">The element type.</typeparam>
    /// <param name="storage">The elements in storage order.</param>
    /// <param name="storageShape">The storage order's shape.</param>
    /// <param name="permutation">The logical axes' storage axes.</param>
    public static T[] ToLogical<T>(ReadOnlySpan<T> storage, IReadOnlyList<long> storageShape, IReadOnlyList<int> permutation)
    {
        ArgumentNullException.ThrowIfNull(storageShape);
        ArgumentNullException.ThrowIfNull(permutation);
        int rank = storageShape.Count;
        var logicalShape = permutation.Select(p => storageShape[p]).ToArray();
        var storageStrides = new long[rank];
        long stride = 1;
        for (int i = rank - 1; i >= 0; i--)
        {
            storageStrides[i] = stride;
            stride *= storageShape[i];
        }

        var result = new T[storage.Length];
        for (long flat = 0; flat < result.Length; flat++)
        {
            long remainder = flat;
            long offset = 0;
            for (int i = rank - 1; i >= 0; i--)
            {
                offset += (remainder % logicalShape[i]) * storageStrides[permutation[i]];
                remainder /= logicalShape[i];
            }

            result[flat] = storage[(int)offset];
        }

        return result;
    }
}
