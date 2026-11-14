using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Architectures;

/// <summary>Graph constants by name: a constant requested again under the same name is the same value (built once).</summary>
internal sealed class ConstantCache(GraphBuilder b)
{
    private readonly Dictionary<string, Value> _values = new(StringComparer.Ordinal);

    /// <summary>An f32 constant of the given shape; <paramref name="data"/> is only called the first time.</summary>
    public Value F32(string name, Shape shape, Func<float[]> data)
    {
        if (!_values.TryGetValue(name, out var v))
        {
            var values = data();
            var bytes = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            v = b.Constant(name, DataType.F32, shape, bytes);
            _values[name] = v;
        }

        return v;
    }

    /// <summary>An i32 vector constant; <paramref name="data"/> is only called the first time.</summary>
    public Value I32(string name, Func<int[]> data)
    {
        if (!_values.TryGetValue(name, out var v))
        {
            var values = data();
            var bytes = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            v = b.Constant(name, DataType.I32, new Shape(values.Length), bytes);
            _values[name] = v;
        }

        return v;
    }
}
