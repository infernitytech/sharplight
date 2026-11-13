using System.Text;
using System.Text.Json;

namespace Sharplight.HuggingFace.Tests;

/// <summary>Writes a single-shard safetensors file from raw tensor bytes (tests only).</summary>
internal sealed class SafetensorsWriter
{
    private readonly List<(string Name, string DType, long[] Shape, byte[] Data)> _tensors = [];

    /// <summary>Adds a tensor with the given safetensors dtype ("F8_E4M3", "BF16", "F32", ...) and raw little-endian bytes.</summary>
    public SafetensorsWriter Add(string name, string dtype, long[] shape, byte[] data)
    {
        _tensors.Add((name, dtype, shape, data));
        return this;
    }

    /// <summary>Adds an F32 tensor.</summary>
    public SafetensorsWriter AddF32(string name, long[] shape, float[] values)
        => Add(name, "F32", shape, [.. values.SelectMany(BitConverter.GetBytes)]);

    /// <summary>Adds a BF16 tensor (values truncated to bfloat16; pass values that are exact in bfloat16).</summary>
    public SafetensorsWriter AddBF16(string name, long[] shape, float[] values)
        => Add(name, "BF16", shape, [.. values.SelectMany(v => BitConverter.GetBytes((ushort)(BitConverter.SingleToUInt32Bits(v) >> 16)))]);

    /// <summary>Writes the file.</summary>
    public void Write(string path)
    {
        var header = new Dictionary<string, object>();
        long offset = 0;
        foreach (var (name, dtype, shape, data) in _tensors)
        {
            header[name] = new Dictionary<string, object> { ["dtype"] = dtype, ["shape"] = shape, ["data_offsets"] = new[] { offset, offset + data.Length } };
            offset += data.Length;
        }

        var json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        using var stream = File.Create(path);
        stream.Write(BitConverter.GetBytes((ulong)json.Length));
        stream.Write(json);
        foreach (var tensor in _tensors)
        {
            stream.Write(tensor.Data);
        }
    }
}
