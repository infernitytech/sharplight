using System.Numerics;
using System.Text;

namespace Sharplight.Import.PyTorch;

/// <summary>Reads the tensor index of a PyTorch checkpoint (zip or legacy stream format) without touching tensor data.</summary>
internal static class TorchCheckpointReader
{
    private const int MaxDepth = 8;

    /// <summary>The name of the only tensor of a file written by <c>torch.save(tensor)</c>.</summary>
    public const string SingleTensorName = "tensor";

    private static readonly BigInteger s_legacyMagic = BigInteger.Parse("119547037146038801333356");

    /// <summary>Indexes the tensors of <paramref name="file"/>.</summary>
    /// <exception cref="UnsafePickleException">The pickle uses a disallowed opcode or global.</exception>
    /// <exception cref="InvalidDataException">The file is malformed.</exception>
    /// <exception cref="NotSupportedException">A tensor is non-contiguous or the byte order is big-endian.</exception>
    public static Dictionary<string, TorchTensorEntry> Read(FileView file, TorchCheckpointOptions options)
    {
        try
        {
            return ReadChecked(file, options);
        }
        catch (OverflowException ex)
        {
            throw new InvalidDataException("A size or offset in the checkpoint overflows.", ex);
        }
    }

    private static Dictionary<string, TorchTensorEntry> ReadChecked(FileView file, TorchCheckpointOptions options)
    {
        if (ZipDirectory.LooksLikeZip(file))
        {
            return ReadZip(file, options);
        }

        if (file.Length > 2 && file.Slice(0, 2) is [0x80, <= 5])
        {
            return ReadLegacy(file, options);
        }

        throw new InvalidDataException("Not a PyTorch checkpoint: neither a zip archive nor a pickle stream.");
    }

    private static Dictionary<string, TorchTensorEntry> ReadZip(FileView file, TorchCheckpointOptions options)
    {
        var zip = ZipDirectory.Read(file);
        var pickleEntries = zip.Entries.Values.Where(e => e.Name == "data.pkl" || e.Name.EndsWith("/data.pkl", StringComparison.Ordinal)).ToList();
        if (pickleEntries.Count != 1)
        {
            throw new InvalidDataException($"Expected exactly one data.pkl in the archive, found {pickleEntries.Count}.");
        }

        var pickleEntry = pickleEntries[0];
        string prefix = pickleEntry.Name[..^"data.pkl".Length];
        if (zip.Entries.TryGetValue(prefix + "byteorder", out var byteOrder)
            && Encoding.ASCII.GetString(file.Slice(byteOrder.DataOffset, (int)Math.Min(byteOrder.Size, 16))) != "little")
        {
            throw new NotSupportedException("Big-endian PyTorch checkpoints are not supported.");
        }

        if (pickleEntry.Size > options.MaxPickleBytes)
        {
            throw new UnsafePickleException($"The pickle stream is {pickleEntry.Size} bytes, more than the limit of {options.MaxPickleBytes}.");
        }

        byte[] pickle = file.Slice(pickleEntry.DataOffset, checked((int)pickleEntry.Size)).ToArray();
        object? root = new RestrictedUnpickler(pickle, 0, options).Load();
        return Resolve(root, options, storage =>
        {
            if (!zip.Entries.TryGetValue($"{prefix}data/{storage.Key}", out var entry))
            {
                throw new InvalidDataException($"Storage '{storage.Key}' is missing from the archive.");
            }

            return (entry.DataOffset, entry.Size);
        });
    }

    private static Dictionary<string, TorchTensorEntry> ReadLegacy(FileView file, TorchCheckpointOptions options)
    {
        // The legacy stream: magic number, protocol version, system info, the object, the list of storage keys, then each storage
        // as an int64 element count followed by its raw elements.
        if (file.Length > int.MaxValue)
        {
            throw new NotSupportedException("Legacy (non-zip) PyTorch checkpoints larger than 2 GB are not supported; re-save with a recent torch.");
        }

        byte[] all = file.Slice(0, (int)file.Length).ToArray();
        int position = 0;
        object? magic = Next(all, ref position, options);
        if (magic is not BigInteger number || number != s_legacyMagic)
        {
            throw new InvalidDataException("Not a PyTorch legacy checkpoint (bad magic number).");
        }

        if (Next(all, ref position, options) is not long protocol || protocol != 1001)
        {
            throw new NotSupportedException("Unsupported PyTorch legacy protocol version.");
        }

        Next(all, ref position, options);
        object? root = Next(all, ref position, options);
        if (Next(all, ref position, options) is not List<object?> keyList)
        {
            throw new InvalidDataException("The legacy checkpoint has no storage key list.");
        }

        var dtypes = new Dictionary<string, TorchDType>(StringComparer.Ordinal);
        CollectStorages(root, dtypes, 0);
        var blobs = new Dictionary<string, (long Offset, long Bytes)>(StringComparer.Ordinal);
        long cursor = position;
        foreach (object? item in keyList)
        {
            if (item is not string key || !dtypes.TryGetValue(key, out var dtype))
            {
                throw new InvalidDataException("The legacy checkpoint lists a storage it never references.");
            }

            long count = file.I64(cursor);
            long bytes = checked(count * dtype.ElementSize);
            if (cursor + 8 + bytes > file.Length)
            {
                throw new InvalidDataException($"Storage '{key}' lies outside the file.");
            }

            blobs[key] = (cursor + 8, bytes);
            cursor += 8 + bytes;
        }

        return Resolve(root, options, storage => blobs.TryGetValue(storage.Key, out var blob)
            ? blob
            : throw new InvalidDataException($"Storage '{storage.Key}' is missing from the file."));
    }

    private static object? Next(byte[] data, ref int position, TorchCheckpointOptions options)
    {
        var unpickler = new RestrictedUnpickler(data, position, options);
        object? value = unpickler.Load();
        position = unpickler.Position;
        return value;
    }

    private static void CollectStorages(object? node, Dictionary<string, TorchDType> dtypes, int depth)
    {
        if (depth > MaxDepth)
        {
            return;
        }

        switch (node)
        {
            case PickleTensorRef tensor:
                dtypes[tensor.Storage.Key] = tensor.Storage.DType;
                break;
            case PickleDict dict:
                foreach (var item in dict.Items)
                {
                    CollectStorages(item.Value, dtypes, depth + 1);
                }

                break;
        }
    }

    private static Dictionary<string, TorchTensorEntry> Resolve(object? root, TorchCheckpointOptions options, Func<PickleStorageRef, (long Offset, long Bytes)> locate)
    {
        if (root is PickleTensorRef single)
        {
            // A file written by torch.save(tensor): one entry named "tensor".
            return new Dictionary<string, TorchTensorEntry>(StringComparer.Ordinal) { [SingleTensorName] = Describe(SingleTensorName, single, locate, options.AllowPermuted) };
        }

        if (root is not PickleDict dict)
        {
            throw new InvalidDataException("The checkpoint does not hold a dictionary of tensors.");
        }

        if (options.StateDictKey is { } key && !dict.Items.Any(i => i.Value is PickleTensorRef) && dict.Find(key) is PickleDict inner)
        {
            dict = inner;
        }

        var tensors = new Dictionary<string, TorchTensorEntry>(StringComparer.Ordinal);
        Flatten(dict, string.Empty, 0, tensors, locate, options.AllowPermuted);
        return tensors;
    }

    private static void Flatten(PickleDict dict, string prefix, int depth, Dictionary<string, TorchTensorEntry> tensors, Func<PickleStorageRef, (long Offset, long Bytes)> locate,
        bool allowPermuted)
    {
        if (depth > MaxDepth)
        {
            return;
        }

        foreach (var (rawKey, value) in dict.Items)
        {
            if (rawKey is not string name)
            {
                continue;
            }

            string full = prefix.Length == 0 ? name : prefix + "." + name;
            switch (value)
            {
                case PickleTensorRef tensor:
                    if (!tensors.TryAdd(full, Describe(full, tensor, locate, allowPermuted)))
                    {
                        throw new InvalidDataException($"Tensor '{full}' appears twice.");
                    }

                    break;
                case PickleDict nested:
                    Flatten(nested, full, depth + 1, tensors, locate, allowPermuted);
                    break;
            }
        }
    }

    private static TorchTensorEntry Describe(string name, PickleTensorRef tensor, Func<PickleStorageRef, (long Offset, long Bytes)> locate, bool allowPermuted)
    {
        long elements = 1;
        foreach (long dim in tensor.Shape)
        {
            elements = checked(elements * dim);
        }

        // Contiguity: dimensions of size one may carry any stride.
        long expected = 1;
        int[]? permutation = null;
        long[] shape = tensor.Shape;
        for (int i = tensor.Shape.Length - 1; i >= 0; i--)
        {
            if (tensor.Shape[i] != 1 && elements > 0 && tensor.Strides[i] != expected)
            {
                permutation = allowPermuted ? DensePermutation(tensor.Shape, tensor.Strides) : null;
                if (permutation is null)
                {
                    throw new NotSupportedException($"Tensor '{name}' is not contiguous (strides [{string.Join(", ", tensor.Strides)}] for shape [{string.Join(", ", tensor.Shape)}]).");
                }

                // Storage order: the axes by decreasing stride; logical axis i is storage axis permutation[i].
                var order = Enumerable.Range(0, tensor.Shape.Length).OrderByDescending(a => tensor.Strides[a]).ThenBy(a => a).ToArray();
                shape = [.. order.Select(a => tensor.Shape[a])];
                break;
            }

            expected = checked(expected * tensor.Shape[i]);
        }

        var (blobOffset, blobBytes) = locate(tensor.Storage);
        int elementSize = tensor.Storage.DType.ElementSize;
        long start = checked(tensor.Storage.ElementOffset + tensor.StorageOffset);
        long bytes = checked(elements * elementSize);
        long begin = checked(start * elementSize);
        if (start + elements > tensor.Storage.ElementCount || begin + bytes > blobBytes)
        {
            throw new InvalidDataException($"Tensor '{name}' extends past its storage.");
        }

        return new TorchTensorEntry(name, tensor.Storage.DType, shape, blobOffset + begin, bytes) { Permutation = permutation };
    }

    // The permutation that makes a dense tensor's strides contiguous (logical axis i = storage axis result[i]), or null when the strides leave gaps.
    private static int[]? DensePermutation(long[] shape, long[] strides)
    {
        var order = Enumerable.Range(0, shape.Length).OrderByDescending(a => strides[a]).ThenBy(a => a).ToArray();
        long expected = 1;
        for (int k = order.Length - 1; k >= 0; k--)
        {
            int axis = order[k];
            if (shape[axis] != 1 && strides[axis] != expected)
            {
                return null;
            }

            expected = checked(expected * shape[axis]);
        }

        var permutation = new int[shape.Length];
        for (int k = 0; k < order.Length; k++)
        {
            permutation[order[k]] = k;
        }

        return permutation;
    }
}
