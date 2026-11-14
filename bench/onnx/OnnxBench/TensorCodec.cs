using System.Runtime.InteropServices;
using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace OnnxBench;

/// <summary>Converts the f32 reference files to and from the graph's tensor types.</summary>
internal static class TensorCodec
{
    public static float[] ReadF32(string path)
    {
        var bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    public static byte[] Encode(float[] values, DataType type)
    {
        if (type == DataType.F32)
        {
            return Bytes(values);
        }

        if (type == DataType.I32)
        {
            return Bytes(values.Select(x => (int)x).ToArray());
        }

        if (type == DataType.I64)
        {
            return Bytes(values.Select(x => (long)x).ToArray());
        }

        if (type == DataType.F16)
        {
            return Bytes(values.Select(x => (Half)x).ToArray());
        }

        if (type == DataType.F64)
        {
            return Bytes(values.Select(x => (double)x).ToArray());
        }

        if (type == DataType.U8 || type == DataType.Bool)
        {
            return values.Select(x => (byte)x).ToArray();
        }

        if (type == DataType.I8)
        {
            return values.Select(x => (byte)(sbyte)x).ToArray();
        }

        throw new NotSupportedException($"cannot feed input type {type}");
    }

    public static float[] ReadOutput(CompiledModel model, string name, Value value, RunParameters run)
    {
        long count = value.Shape.Elements(run.Symbols);
        var raw = new byte[(int)value.Type.ByteSize(count)];
        model.GetOutput(name, raw);
        var result = new float[count];
        for (int i = 0; i < count; i++)
        {
            result[i] = Decode(raw, i, value.Type);
        }

        return result;
    }

    private static float Decode(byte[] raw, int i, DataType type)
    {
        if (type == DataType.F32)
        {
            return BitConverter.ToSingle(raw, i * 4);
        }

        if (type == DataType.I32)
        {
            return BitConverter.ToInt32(raw, i * 4);
        }

        if (type == DataType.I64)
        {
            return BitConverter.ToInt64(raw, i * 8);
        }

        if (type == DataType.F16)
        {
            return (float)BitConverter.ToHalf(raw, i * 2);
        }

        if (type == DataType.F64)
        {
            return (float)BitConverter.ToDouble(raw, i * 8);
        }

        if (type == DataType.U8 || type == DataType.Bool)
        {
            return raw[i];
        }

        if (type == DataType.I8)
        {
            return (sbyte)raw[i];
        }

        throw new NotSupportedException($"cannot read output type {type}");
    }

    private static byte[] Bytes<T>(T[] values) where T : unmanaged => MemoryMarshal.AsBytes(values.AsSpan()).ToArray();
}
