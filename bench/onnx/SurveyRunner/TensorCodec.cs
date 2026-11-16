using Sharplight.Graph;
using Sharplight.Runtime;
using Sharplight.Tensors;

/// <summary>Converts between the reference f32 files and the tensor element types of the graph.</summary>
internal static class TensorCodec
{
    /// <summary>Reads a raw little-endian f32 file.</summary>
    public static float[] ReadF32(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var values = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
        return values;
    }

    /// <summary>Writes <paramref name="values"/> as raw f32.</summary>
    public static void WriteF32(string path, float[] values)
    {
        File.WriteAllBytes(path, ToBytes(values));
    }

    /// <summary>Converts f32 reference values to the bytes of an input of element type <paramref name="type"/>.</summary>
    public static byte[] FromF32(float[] values, DataType type)
    {
        if (type == DataType.F32)
        {
            return ToBytes(values);
        }

        if (type == DataType.I32)
        {
            return ToBytes(values.Select(x => (int)x).ToArray(), 4);
        }

        if (type == DataType.I64)
        {
            return ToBytes(values.Select(x => (long)x).ToArray(), 8);
        }

        if (type == DataType.F16)
        {
            return ToBytes(values.Select(x => (Half)x).ToArray(), 2);
        }

        if (type == DataType.F64)
        {
            return ToBytes(values.Select(x => (double)x).ToArray(), 8);
        }

        if (type == DataType.U8 || type == DataType.Bool)
        {
            return values.Select(x => (byte)x).ToArray();
        }

        if (type == DataType.I8)
        {
            return values.Select(x => (byte)(sbyte)x).ToArray();
        }

        throw new NotSupportedException($"harness cannot feed input type {type}");
    }

    /// <summary>Reads the output <paramref name="name"/> of <paramref name="model"/> as f32.</summary>
    public static float[] ReadOutput(CompiledModel model, string name, Value value, RunParameters run)
    {
        long count = value.Shape.Elements(run.Symbols);
        var raw = new byte[(int)value.Type.ByteSize(count)];
        model.GetOutput(name, raw);
        var output = new float[count];
        var type = value.Type;
        if (type == DataType.F32)
        {
            Buffer.BlockCopy(raw, 0, output, 0, raw.Length);
        }
        else if (type == DataType.I32)
        {
            Convert(output, i => BitConverter.ToInt32(raw, i * 4));
        }
        else if (type == DataType.I64)
        {
            Convert(output, i => BitConverter.ToInt64(raw, i * 8));
        }
        else if (type == DataType.F16)
        {
            Convert(output, i => (float)BitConverter.ToHalf(raw, i * 2));
        }
        else if (type == DataType.F64)
        {
            Convert(output, i => (float)BitConverter.ToDouble(raw, i * 8));
        }
        else if (type == DataType.U8 || type == DataType.Bool)
        {
            Convert(output, i => raw[i]);
        }
        else if (type == DataType.I8)
        {
            Convert(output, i => (sbyte)raw[i]);
        }
        else
        {
            throw new NotSupportedException($"harness cannot read output type {type}");
        }

        return output;
    }

    private static void Convert(float[] output, Func<int, float> element)
    {
        for (int i = 0; i < output.Length; i++)
        {
            output[i] = element(i);
        }
    }

    private static byte[] ToBytes(float[] values)
    {
        return ToBytes(values, 4);
    }

    private static byte[] ToBytes(Array values, int elementSize)
    {
        var bytes = new byte[values.Length * elementSize];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
