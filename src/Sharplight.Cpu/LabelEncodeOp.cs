using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>
/// Elementwise hash lookup of numeric or string keys to numeric or string values (see <c>StandardOps.LabelEncode</c>). Keys and values are
/// held as objects (double or string) so that every key / value type combination shares one kernel.
/// </summary>
internal sealed unsafe class LabelEncodeOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly Dictionary<object, object> _map = [];
    private readonly object _default;
    private readonly object? _nanValue;

    public LabelEncodeOp(TRef x, TRef y, object[] keys, object[] values, object fallback)
    {
        _x = x;
        _y = y;
        _default = fallback;
        for (int i = 0; i < keys.Length; i++)
        {
            if (keys[i] is double.NaN)
            {
                _nanValue = values[i];
            }
            else
            {
                // f32 inputs carry float keys (double tensors are rounded the same way).
                _map.TryAdd(keys[i] is double d && x.Type == DataType.F32 ? (double)(float)d : keys[i], values[i]);
            }
        }
    }

    public override string Name => "LabelEncode";

    public override void Run(RunParameters p)
    {
        long n = _x.Elements(p);
        for (long i = 0; i < n; i++)
        {
            object key = _x.Type == DataType.Str ? p.Strings[((int*)_x.P)[i]]
                : _x.Type == DataType.I32 ? (double)((int*)_x.P)[i]
                : (double)((float*)_x.P)[i];
            object value = key is double.NaN ? _nanValue ?? _default : _map.GetValueOrDefault(key, _default);
            if (_y.Type == DataType.Str)
            {
                ((int*)_y.P)[i] = p.Strings.Add((string)value);
            }
            else if (_y.Type == DataType.I32)
            {
                ((int*)_y.P)[i] = (int)(double)value;
            }
            else
            {
                ((float*)_y.P)[i] = (float)(double)value;
            }
        }
    }
}
