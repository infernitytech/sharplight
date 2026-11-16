namespace Sharplight.Import.HuggingFace;

/// <summary>Builds the tensor table a quantization reader hands on: the unconsumed originals plus the converted tensors.</summary>
internal static class HfTensorMerge
{
    /// <summary>
    /// The tensors of <paramref name="tensors"/> that are not in <paramref name="consumed"/>, followed by
    /// <paramref name="converted"/> (which wins on a name clash).
    /// </summary>
    public static Dictionary<string, SafetensorsTensor> Replace(
        IReadOnlyDictionary<string, SafetensorsTensor> tensors,
        HashSet<string> consumed,
        IEnumerable<KeyValuePair<string, SafetensorsTensor>> converted)
    {
        var result = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in tensors)
        {
            if (!consumed.Contains(name))
            {
                result[name] = tensor;
            }
        }

        foreach (var (name, tensor) in converted)
        {
            result[name] = tensor;
        }

        return result;
    }
}
