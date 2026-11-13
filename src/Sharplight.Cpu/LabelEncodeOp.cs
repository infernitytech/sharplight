using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Cpu;

/// <summary>Elementwise hash lookup of numeric keys (see <c>StandardOps.LabelEncode</c>).</summary>
internal sealed unsafe class LabelEncodeOp : CpuOp
{
    private readonly TRef _x;
    private readonly TRef _y;
    private readonly Dictionary<double, double> _map = [];
    private readonly double _default;
    private readonly double? _nanValue;

    public LabelEncodeOp(TRef x, TRef y, double[] keys, double[] values, double fallback)
    {
        _x = x;
        _y = y;
        _default = fallback;
        for (int i = 0; i < keys.Length; i++)
        {
            if (double.IsNaN(keys[i]))
            {
                _nanValue = values[i];
            }
            else
            {
                // f32 inputs carry float keys (double tensors are rounded the same way).
                _map.TryAdd(x.Type == DataType.I32 ? keys[i] : (float)keys[i], values[i]);
            }
        }
    }

    public override string Name => "LabelEncode";

    public override void Run(RunParameters p)
    {
        long n = _x.Elements(p);
        bool intIn = _x.Type == DataType.I32;
        bool intOut = _y.Type == DataType.I32;
        for (long i = 0; i < n; i++)
        {
            double key = intIn ? ((int*)_x.P)[i] : ((float*)_x.P)[i];
            double value = double.IsNaN(key) ? _nanValue ?? _default : _map.TryGetValue(key, out var v) ? v : _default;
            if (intOut)
            {
                ((int*)_y.P)[i] = (int)value;
            }
            else
            {
                ((float*)_y.P)[i] = (float)value;
            }
        }
    }
}
