using Sharplight.Tensors;

namespace Sharplight.Graph.Optimization;

/// <summary>
/// Host evaluation of single-output ops over constant operands, bit-exact with the reference CPU backend. Only ops whose CPU
/// semantics are reproduced exactly are supported: data movement (any plain element type), casts, <c>Fill</c>, <c>Where</c>, and
/// f32 elementwise ops that are correctly rounded or exact (no transcendental functions, whose CPU kernels use vectorized
/// approximations). Anything else, and any case whose result could depend on the kernel path (NaN or signed-zero operands of
/// <c>Max</c>/<c>Min</c>, out-of-range indices), is refused with <see langword="null"/>.
/// </summary>
internal static class ConstantEvaluator
{
    private static readonly HashSet<string> s_unary =
        ["Neg", "Abs", "Relu", "Square", "Recip", "Sqrt", "Rsqrt", "Floor", "Ceil", "Round", "Sign", "Not", "IsNaN", "IsInf"];
    private static readonly HashSet<string> s_binary =
        ["Add", "Sub", "Mul", "Div", "Max", "Min", "Mod", "PRelu", "Equal", "NotEqual", "Less", "LessEqual", "Greater", "GreaterEqual", "And", "Or", "Xor"];
    private static readonly HashSet<string> s_movement = ["Reshape", "Contiguous", "Permute", "Slice", "Concat", "Expand", "Tile", "GatherAxis"];

    /// <summary>Whether the op is one the evaluator may be able to fold.</summary>
    public static bool Supports(string op) => s_unary.Contains(op) || s_binary.Contains(op) || s_movement.Contains(op) || op is "Cast" or "Fill" or "Where";

    /// <summary>
    /// Evaluates a single-output node whose operands are <paramref name="inputs"/> into an output of the static shape
    /// <paramref name="dims"/>.
    /// </summary>
    /// <returns>The output bytes, or <see langword="null"/> when the case is not supported exactly.</returns>
    public static byte[]? Evaluate(Node n, IReadOnlyList<HostTensor> inputs, long[] dims)
    {
        string op = n.Op.Name;
        var type = n.Output.Type;
        long elements = dims.Aggregate(1L, (a, b) => a * b);
        if (s_unary.Contains(op))
        {
            return HostTensor.IsFloatLike(inputs[0].Type) && type == DataType.F32 ? Unary(op, inputs[0], elements) : null;
        }

        if (s_binary.Contains(op))
        {
            return inputs.All(i => HostTensor.IsFloatLike(i.Type)) && type == DataType.F32 ? Binary(op, inputs[0], inputs[1], dims, n.Attributes.Get("fmod", false)) : null;
        }

        if (op == "Cast")
        {
            return Cast(inputs[0], type);
        }

        if (op == "Fill")
        {
            return Fill(type, dims, n.Attributes.Get("start", 0.0), n.Attributes.Get("delta", 0.0));
        }

        if (op == "Where")
        {
            return Where(inputs, type, dims);
        }

        // Data movement: the moved operands (all but GatherAxis's indices) have the output's plain element type.
        var moved = op == "GatherAxis" ? inputs.Take(1) : inputs;
        if (!IsPlain(type) || moved.Any(i => i.Type != type))
        {
            return null;
        }

        return op switch
        {
            "Reshape" or "Contiguous" => (byte[])inputs[0].Data.Clone(),
            "Permute" => Permute(inputs[0], n.Attributes.Get<int[]>("perm"), dims),
            "Slice" => Slice(inputs[0], Axis(n.Attributes.Get<long>("axis"), inputs[0].Dims.Length), n.Attributes.Get<long>("start"), dims),
            "Concat" => Concat(inputs, Axis(n.Attributes.Get<long>("axis"), dims.Length), dims),
            "Expand" => Broadcast(inputs[0], dims),
            "Tile" => Tile(inputs[0], dims),
            "GatherAxis" => GatherAxis(inputs[0], inputs[1], Axis(n.Attributes.Get("axis", 0L), inputs[0].Dims.Length), dims),
            _ => null,
        };
    }

    private static bool IsPlain(DataType t) => !t.IsQuantized && t.BlockSize == 1 && t.BlockBytes is 1 or 2 or 4 or 8;

    private static int Axis(long axis, int rank) => (int)(axis < 0 ? axis + rank : axis);

    private static long[] RowMajorStrides(long[] dims)
    {
        var strides = new long[dims.Length];
        long s = 1;
        for (int k = dims.Length - 1; k >= 0; k--)
        {
            strides[k] = s;
            s *= dims[k];
        }

        return strides;
    }

    /// <summary>Element strides of <paramref name="inDims"/> broadcast (numpy rules, aligned right) to <paramref name="outDims"/>; 0 on broadcast axes.</summary>
    private static long[]? BroadcastStrides(long[] inDims, long[] outDims)
    {
        if (inDims.Length > outDims.Length)
        {
            return null;
        }

        var own = RowMajorStrides(inDims);
        var strides = new long[outDims.Length];
        int offset = outDims.Length - inDims.Length;
        for (int k = 0; k < inDims.Length; k++)
        {
            long d = inDims[k];
            if (d != outDims[k + offset] && d != 1)
            {
                return null;
            }

            strides[k + offset] = d == 1 ? 0 : own[k];
        }

        return strides;
    }

    /// <summary>Source element offset of output element <paramref name="flat"/> given per-axis source strides.</summary>
    private static long Offset(long flat, long[] dims, long[] strides)
    {
        long offset = 0;
        for (int k = dims.Length - 1; k >= 0; k--)
        {
            long index = flat % dims[k];
            flat /= dims[k];
            offset += index * strides[k];
        }

        return offset;
    }

    private static byte[] Floats(long elements, Func<long, float> value)
    {
        var data = new byte[elements * 4];
        var span = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(data.AsSpan());
        for (long i = 0; i < elements; i++)
        {
            span[(int)i] = value(i);
        }

        return data;
    }

    // Mirrors Sharplight.Cpu ScalarMath.Unary for the exact kinds.
    private static byte[] Unary(string op, HostTensor x, long elements) => Floats(elements, i =>
    {
        float v = x.ReadFloat(i);
        return op switch
        {
            "Neg" => -v,
            "Abs" => MathF.Abs(v),
            "Relu" => v > 0 ? v : 0,
            "Square" => v * v,
            "Recip" => 1f / v,
            "Sqrt" => MathF.Sqrt(v),
            "Rsqrt" => 1f / MathF.Sqrt(v),
            "Floor" => MathF.Floor(v),
            "Ceil" => MathF.Ceiling(v),
            "Round" => MathF.Round(v, MidpointRounding.ToEven),
            "Sign" => v > 0 ? 1f : v < 0 ? -1f : v,
            "Not" => v != 0 ? 0f : 1f,
            "IsNaN" => float.IsNaN(v) ? 1f : 0f,
            _ => float.IsInfinity(v) ? 1f : 0f,
        };
    });

    // Mirrors Sharplight.Cpu BinaryOp (OpS / OpS2).
    private static byte[]? Binary(string op, HostTensor a, HostTensor b, long[] dims, bool fmod)
    {
        var sa = BroadcastStrides(a.Dims, dims);
        var sb = BroadcastStrides(b.Dims, dims);
        if (sa is null || sb is null)
        {
            return null;
        }

        long elements = dims.Aggregate(1L, (x, y) => x * y);
        bool exact = true;
        var data = Floats(elements, i =>
        {
            float x = a.ReadFloat(Offset(i, dims, sa));
            float y = b.ReadFloat(Offset(i, dims, sb));
            if (op is "Max" or "Min" && (float.IsNaN(x) || float.IsNaN(y) || (x == 0 && y == 0 && float.IsNegative(x) != float.IsNegative(y))))
            {
                exact = false; // the vectorized and scalar kernels differ here
            }

            return Apply(op, x, y, fmod);
        });
        return exact ? data : null;
    }

    private static float Apply(string op, float a, float b, bool fmod)
    {
        switch (op)
        {
            case "Add":
                return a + b;
            case "Sub":
                return a - b;
            case "Mul":
                return a * b;
            case "Div":
                return a / b;
            case "Max":
                return MathF.Max(a, b);
            case "Min":
                return MathF.Min(a, b);
            case "Mod":
            {
                float r = a % b;
                if (!fmod && r != 0 && (r < 0) != (b < 0))
                {
                    r += b;
                }

                return r;
            }
            case "PRelu":
                return a < 0 ? a * b : a;
            case "Equal":
                return a == b ? 1f : 0f;
            case "NotEqual":
                return a != b ? 1f : 0f;
            case "Less":
                return a < b ? 1f : 0f;
            case "LessEqual":
                return a <= b ? 1f : 0f;
            case "Greater":
                return a > b ? 1f : 0f;
            case "GreaterEqual":
                return a >= b ? 1f : 0f;
            case "And":
                return a != 0 && b != 0 ? 1f : 0f;
            case "Or":
                return a != 0 || b != 0 ? 1f : 0f;
            default:
                return (a != 0) != (b != 0) ? 1f : 0f;
        }
    }

    private static readonly DataType[] s_castSources = [DataType.F32, DataType.F16, DataType.BF16, DataType.I32, DataType.I64, DataType.I8, DataType.U8, DataType.Bool];
    private static readonly DataType[] s_castTargets = [DataType.F32, DataType.I32, DataType.I64, DataType.I8, DataType.U8, DataType.Bool];

    // Mirrors Sharplight.Cpu CastOp: direct paths for f32→f32, i32→i32, f16/bf16→f32, i32↔i64; everything else through f32.
    private static byte[]? Cast(HostTensor x, DataType target)
    {
        var source = x.Type;
        if (!s_castSources.Contains(source) || !s_castTargets.Contains(target))
        {
            return null;
        }

        long n = x.Elements;
        var output = new byte[target.ByteSize(n)];
        var src = x.Data.AsSpan();
        var dst = output.AsSpan();
        if (source == target && (source == DataType.F32 || source == DataType.I32))
        {
            src.CopyTo(dst);
            return output;
        }

        for (int i = 0; i < n; i++)
        {
            if (source == DataType.I32 && target == DataType.I64)
            {
                BitConverter.TryWriteBytes(dst[(i * 8)..], (long)BitConverter.ToInt32(src[(i * 4)..]));
            }
            else if (source == DataType.I64 && target == DataType.I32)
            {
                BitConverter.TryWriteBytes(dst[(i * 4)..], (int)BitConverter.ToInt64(src[(i * 8)..]));
            }
            else
            {
                Store(target, dst, i, Load(x, i));
            }
        }

        return output;
    }

    private static float Load(HostTensor x, int i)
    {
        var t = x.Type;
        var d = x.Data;
        if (HostTensor.IsFloatLike(t))
        {
            return x.ReadFloat(i);
        }

        if (t == DataType.I32)
        {
            return BitConverter.ToInt32(d, i * 4);
        }

        if (t == DataType.I64)
        {
            return BitConverter.ToInt64(d, i * 8);
        }

        if (t == DataType.I8)
        {
            return (sbyte)d[i];
        }

        if (t == DataType.U8)
        {
            return d[i];
        }

        return d[i] != 0 ? 1 : 0;
    }

    private static void Store(DataType t, Span<byte> d, int i, float v)
    {
        if (t == DataType.F32)
        {
            BitConverter.TryWriteBytes(d[(i * 4)..], v);
        }
        else if (t == DataType.I32)
        {
            BitConverter.TryWriteBytes(d[(i * 4)..], (int)v);
        }
        else if (t == DataType.I64)
        {
            BitConverter.TryWriteBytes(d[(i * 8)..], (long)v);
        }
        else if (t == DataType.I8)
        {
            d[i] = unchecked((byte)(sbyte)v);
        }
        else if (t == DataType.U8)
        {
            d[i] = (byte)v;
        }
        else
        {
            d[i] = (byte)(v != 0 ? 1 : 0);
        }
    }

    // Mirrors Sharplight.Cpu FillOp.
    private static byte[]? Fill(DataType type, long[] dims, double start, double delta)
    {
        if (type != DataType.F32 && type != DataType.I32 && type != DataType.I64)
        {
            return null;
        }

        long rows = dims.Length == 0 ? 1 : dims[0];
        long elements = dims.Aggregate(1L, (a, b) => a * b);
        long inner = dims.Length == 0 ? 1 : elements / Math.Max(1, dims[0]);
        var data = new byte[type.ByteSize(elements)];
        var span = data.AsSpan();
        for (long r = 0; r < rows; r++)
        {
            double v = start + r * delta;
            for (long k = 0; k < inner; k++)
            {
                int i = (int)(r * inner + k);
                if (type == DataType.I32)
                {
                    BitConverter.TryWriteBytes(span[(i * 4)..], (int)v);
                }
                else if (type == DataType.I64)
                {
                    BitConverter.TryWriteBytes(span[(i * 8)..], (long)v);
                }
                else
                {
                    BitConverter.TryWriteBytes(span[(i * 4)..], (float)v);
                }
            }
        }

        return data;
    }

    // Mirrors Sharplight.Cpu WhereOp: any non-zero condition element selects the first branch.
    private static byte[]? Where(IReadOnlyList<HostTensor> inputs, DataType type, long[] dims)
    {
        var (c, a, b) = (inputs[0], inputs[1], inputs[2]);
        bool conditionOk = c.Type == DataType.F32 || c.Type == DataType.I32 || c.Type == DataType.I64 || c.Type == DataType.U8 || c.Type == DataType.Bool;
        if (!conditionOk || a.Type != type || b.Type != type || !IsPlain(type))
        {
            return null;
        }

        var sc = BroadcastStrides(c.Dims, dims);
        var sa = BroadcastStrides(a.Dims, dims);
        var sb = BroadcastStrides(b.Dims, dims);
        if (sc is null || sa is null || sb is null)
        {
            return null;
        }

        int es = type.BlockBytes;
        long elements = dims.Aggregate(1L, (x, y) => x * y);
        var output = new byte[elements * es];
        for (long i = 0; i < elements; i++)
        {
            long oc = Offset(i, dims, sc);
            bool take = c.Type == DataType.F32 ? BitConverter.ToSingle(c.Data, (int)(oc * 4)) != 0f
                : c.Type == DataType.I32 ? BitConverter.ToInt32(c.Data, (int)(oc * 4)) != 0
                : c.Type == DataType.I64 ? BitConverter.ToInt64(c.Data, (int)(oc * 8)) != 0
                : c.Data[oc] != 0;
            var (from, strides) = take ? (a, sa) : (b, sb);
            Array.Copy(from.Data, Offset(i, dims, strides) * es, output, i * es, es);
        }

        return output;
    }

    private static byte[] Gather(HostTensor x, long[] dims, Func<long, long> sourceElement)
    {
        int es = x.Type.BlockBytes;
        long elements = dims.Aggregate(1L, (a, b) => a * b);
        var output = new byte[elements * es];
        for (long i = 0; i < elements; i++)
        {
            Array.Copy(x.Data, sourceElement(i) * es, output, i * es, es);
        }

        return output;
    }

    private static byte[]? Permute(HostTensor x, int[] perm, long[] dims)
    {
        if (perm.Length != x.Dims.Length)
        {
            return null;
        }

        var own = RowMajorStrides(x.Dims);
        var strides = perm.Select(p => own[p]).ToArray();
        return Gather(x, dims, i => Offset(i, dims, strides));
    }

    private static byte[]? Slice(HostTensor x, int axis, long start, long[] dims)
    {
        if (axis < 0 || axis >= x.Dims.Length || start < 0 || start + dims[axis] > x.Dims[axis])
        {
            return null;
        }

        var strides = RowMajorStrides(x.Dims);
        long baseOffset = start * strides[axis];
        return Gather(x, dims, i => baseOffset + Offset(i, dims, strides));
    }

    private static byte[]? Broadcast(HostTensor x, long[] dims)
    {
        var strides = BroadcastStrides(x.Dims, dims);
        return strides is null ? null : Gather(x, dims, i => Offset(i, dims, strides));
    }

    private static byte[]? Tile(HostTensor x, long[] dims)
    {
        if (x.Dims.Length != dims.Length)
        {
            return null;
        }

        var strides = RowMajorStrides(x.Dims);
        return Gather(x, dims, i =>
        {
            long offset = 0;
            for (int k = dims.Length - 1; k >= 0; k--)
            {
                long index = i % dims[k];
                i /= dims[k];
                offset += index % x.Dims[k] * strides[k];
            }

            return offset;
        });
    }

    private static byte[]? Concat(IReadOnlyList<HostTensor> inputs, int axis, long[] dims)
    {
        if (axis < 0 || axis >= dims.Length || inputs.Any(t => t.Dims.Length != dims.Length))
        {
            return null;
        }

        long outer = dims.Take(axis).Aggregate(1L, (a, b) => a * b);
        long inner = dims.Skip(axis + 1).Aggregate(1L, (a, b) => a * b);
        int es = inputs[0].Type.BlockBytes;
        var output = new byte[outer * dims[axis] * inner * es];
        long position = 0;
        foreach (var t in inputs)
        {
            long chunk = t.Dims[axis] * inner * es;
            for (long o = 0; o < outer; o++)
            {
                Array.Copy(t.Data, o * chunk, output, (o * dims[axis] * inner + position * inner) * es, chunk);
            }

            position += t.Dims[axis];
        }

        return position == dims[axis] ? output : null;
    }

    // Mirrors Sharplight.Cpu GatherAxisOp: negative indices count from the end; out-of-range indices fail at run time, so they are not folded.
    private static byte[]? GatherAxis(HostTensor x, HostTensor indices, int axis, long[] dims)
    {
        if ((indices.Type != DataType.I32 && indices.Type != DataType.I64) || axis < 0 || axis >= x.Dims.Length)
        {
            return null;
        }

        long extent = x.Dims[axis];
        long outer = x.Dims.Take(axis).Aggregate(1L, (a, b) => a * b);
        long inner = x.Dims.Skip(axis + 1).Aggregate(1L, (a, b) => a * b);
        long q = indices.Elements;
        var rows = new long[q];
        for (int k = 0; k < q; k++)
        {
            long id = indices.Type == DataType.I64 ? BitConverter.ToInt64(indices.Data, k * 8) : BitConverter.ToInt32(indices.Data, k * 4);
            if (id < 0)
            {
                id += extent;
            }

            if (id < 0 || id >= extent)
            {
                return null;
            }

            rows[k] = id;
        }

        if (outer * q * inner != dims.Aggregate(1L, (a, b) => a * b))
        {
            return null;
        }

        return Gather(x, dims, i =>
        {
            long j = i % inner;
            long r = i / inner;
            long k = r % q;
            long o = r / q;
            return (o * extent + rows[k]) * inner + j;
        });
    }
}
