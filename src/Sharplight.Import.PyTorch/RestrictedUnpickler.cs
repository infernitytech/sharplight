using System.Buffers.Binary;
using System.Numerics;
using System.Text;

namespace Sharplight.Import.PyTorch;

/// <summary>
/// A pickle virtual machine for tensor state dictionaries. It implements only the opcodes such files use, resolves only an
/// allow-list of globals to behavior implemented in this class (never by reflection or dynamic loading), and refuses everything
/// else with an <see cref="UnsafePickleException"/>. It never calls into pickled code: <c>REDUCE</c> dispatches on the closed
/// <see cref="PickleGlobal"/> set, <c>BUILD</c> only stores attributes on dictionaries or inert placeholders.
/// </summary>
internal sealed class RestrictedUnpickler
{
    private const int MaxDims = 16;

    private readonly ReadOnlyMemory<byte> _memory;
    private readonly TorchCheckpointOptions _options;
    private readonly List<object?> _stack = [];
    private readonly List<int> _marks = [];
    private readonly Dictionary<long, object?> _memo = [];
    private int _position;

    /// <summary>Creates an unpickler over <paramref name="data"/> starting at <paramref name="position"/>.</summary>
    public RestrictedUnpickler(ReadOnlyMemory<byte> data, int position, TorchCheckpointOptions options)
    {
        _memory = data;
        _position = position;
        _options = options;
    }

    /// <summary>The offset just after the last opcode read (after the STOP opcode once <see cref="Load"/> returned).</summary>
    public int Position => _position;

    /// <summary>Runs the pickle up to its STOP opcode and returns the resulting object (dictionaries as <see cref="PickleDict"/>).</summary>
    /// <exception cref="UnsafePickleException">The stream uses a disallowed opcode or global, is truncated, or exceeds a limit.</exception>
    public object? Load()
    {
        var data = _memory.Span;
        if (data.Length > _options.MaxPickleBytes)
        {
            throw new UnsafePickleException($"The pickle stream is {data.Length} bytes, more than the limit of {_options.MaxPickleBytes}.");
        }

        while (true)
        {
            byte op = ReadByte();
            switch (op)
            {
                case PickleOp.Proto:
                    {
                        byte version = ReadByte();
                        if (version > 5)
                        {
                            throw new UnsafePickleException($"Unsupported pickle protocol {version}.");
                        }

                        break;
                    }

                case PickleOp.Frame:
                    Take(8);
                    break;
                case PickleOp.Stop:
                    return Pop();
                case PickleOp.Mark:
                    _marks.Add(_stack.Count);
                    break;
                case PickleOp.Pop:
                    Pop();
                    break;
                case PickleOp.PopMark:
                    {
                        int mark = PopMark();
                        _stack.RemoveRange(mark, _stack.Count - mark);
                        break;
                    }

                case PickleOp.Dup:
                    {
                        if (_stack.Count == 0)
                        {
                            throw Malformed("DUP on an empty stack");
                        }

                        Push(_stack[^1]);
                        break;
                    }

                case PickleOp.None:
                    Push(null);
                    break;
                case PickleOp.NewTrue:
                    Push(true);
                    break;
                case PickleOp.NewFalse:
                    Push(false);
                    break;
                case PickleOp.BinInt:
                    Push((long)BinaryPrimitives.ReadInt32LittleEndian(Take(4)));
                    break;
                case PickleOp.BinInt1:
                    Push((long)ReadByte());
                    break;
                case PickleOp.BinInt2:
                    Push((long)BinaryPrimitives.ReadUInt16LittleEndian(Take(2)));
                    break;
                case PickleOp.Long1:
                    Push(ReadLongBytes(ReadByte()));
                    break;
                case PickleOp.Long4:
                    Push(ReadLongBytes(BinaryPrimitives.ReadInt32LittleEndian(Take(4))));
                    break;
                case PickleOp.Int:
                    Push(ParseTextInt(ReadLine()));
                    break;
                case PickleOp.Long:
                    Push(ParseTextLong(ReadLine()));
                    break;
                case PickleOp.BinFloat:
                    Push(BinaryPrimitives.ReadDoubleBigEndian(Take(8)));
                    break;
                case PickleOp.ShortBinUnicode:
                    Push(Encoding.UTF8.GetString(Take(ReadByte())));
                    break;
                case PickleOp.BinUnicode:
                    Push(Encoding.UTF8.GetString(Take(ReadLength32())));
                    break;
                case PickleOp.BinUnicode8:
                    Push(Encoding.UTF8.GetString(Take(ReadLength64())));
                    break;
                case PickleOp.ShortBinString:
                    Push(Encoding.Latin1.GetString(Take(ReadByte())));
                    break;
                case PickleOp.BinString:
                    Push(Encoding.Latin1.GetString(Take(ReadLength32())));
                    break;
                case PickleOp.ShortBinBytes:
                    Push(Take(ReadByte()).ToArray());
                    break;
                case PickleOp.BinBytes:
                    Push(Take(ReadLength32()).ToArray());
                    break;
                case PickleOp.BinBytes8:
                    Push(Take(ReadLength64()).ToArray());
                    break;
                case PickleOp.EmptyTuple:
                    Push(Array.Empty<object?>());
                    break;
                case PickleOp.Tuple1:
                    PushTuple(1);
                    break;
                case PickleOp.Tuple2:
                    PushTuple(2);
                    break;
                case PickleOp.Tuple3:
                    PushTuple(3);
                    break;
                case PickleOp.Tuple:
                    {
                        int mark = PopMark();
                        var items = _stack.GetRange(mark, _stack.Count - mark).ToArray();
                        _stack.RemoveRange(mark, _stack.Count - mark);
                        Push(items);
                        break;
                    }

                case PickleOp.EmptyList:
                    Push(new List<object?>());
                    break;
                case PickleOp.List:
                    {
                        int mark = PopMark();
                        var items = _stack.GetRange(mark, _stack.Count - mark);
                        _stack.RemoveRange(mark, _stack.Count - mark);
                        Push(items);
                        break;
                    }

                case PickleOp.Append:
                    {
                        object? value = Pop();
                        AppendTo(PeekTop(), [value]);
                        break;
                    }

                case PickleOp.Appends:
                    {
                        int mark = PopMark();
                        var items = _stack.GetRange(mark, _stack.Count - mark);
                        _stack.RemoveRange(mark, _stack.Count - mark);
                        AppendTo(PeekTop(), items);
                        break;
                    }

                case PickleOp.EmptyDict:
                    Push(new PickleDict());
                    break;
                case PickleOp.Dict:
                    {
                        int mark = PopMark();
                        var dict = new PickleDict();
                        SetItems(dict, _stack.GetRange(mark, _stack.Count - mark));
                        _stack.RemoveRange(mark, _stack.Count - mark);
                        Push(dict);
                        break;
                    }

                case PickleOp.SetItem:
                    {
                        object? value = Pop();
                        object? key = Pop();
                        SetItems(PeekTop(), [key, value]);
                        break;
                    }

                case PickleOp.SetItems:
                    {
                        int mark = PopMark();
                        var items = _stack.GetRange(mark, _stack.Count - mark);
                        _stack.RemoveRange(mark, _stack.Count - mark);
                        SetItems(PeekTop(), items);
                        break;
                    }

                case PickleOp.BinPut:
                    Memoize(ReadByte());
                    break;
                case PickleOp.LongBinPut:
                    Memoize(BinaryPrimitives.ReadUInt32LittleEndian(Take(4)));
                    break;
                case PickleOp.Memoize:
                    Memoize(_memo.Count);
                    break;
                case PickleOp.BinGet:
                    Push(MemoGet(ReadByte()));
                    break;
                case PickleOp.LongBinGet:
                    Push(MemoGet(BinaryPrimitives.ReadUInt32LittleEndian(Take(4))));
                    break;
                case PickleOp.Global:
                    {
                        string module = ReadLine();
                        string name = ReadLine();
                        Push(ResolveGlobal(module, name));
                        break;
                    }

                case PickleOp.StackGlobal:
                    {
                        object? name = Pop();
                        object? module = Pop();
                        if (module is not string moduleName || name is not string globalName)
                        {
                            throw Malformed("STACK_GLOBAL needs two strings");
                        }

                        Push(ResolveGlobal(moduleName, globalName));
                        break;
                    }

                case PickleOp.Reduce:
                    {
                        object? args = Pop();
                        object? callable = Pop();
                        Push(Call(callable, args as object?[] ?? throw Malformed("REDUCE arguments are not a tuple")));
                        break;
                    }

                case PickleOp.NewObj:
                    {
                        object? args = Pop();
                        object? cls = Pop();
                        if (args is not object?[])
                        {
                            throw Malformed("NEWOBJ arguments are not a tuple");
                        }

                        Push(cls switch
                        {
                            PickleGlobal.OrderedDict => new PickleDict(),
                            PickleInert => PickleInert.Instance,
                            _ => throw Refused("opcode NEWOBJ on a class that is not an inert placeholder"),
                        });
                        break;
                    }

                case PickleOp.Build:
                    {
                        // An inert placeholder discards any state (e.g. numpy dtypes build from a tuple); a dictionary only takes dictionaries.
                        object? state = Pop();
                        object? target = PeekTop();
                        if (target is not PickleInert && (target is not PickleDict || state is not (PickleDict or PickleInert or null)))
                        {
                            throw Refused("opcode BUILD on an object that is not a dictionary or an inert placeholder");
                        }

                        break;
                    }

                case PickleOp.BinPersId:
                    Push(ResolvePersistentId(Pop()));
                    break;
                default:
                    throw Refused($"opcode {PickleOp.Describe(op)}");
            }

            if (_stack.Count > _options.MaxObjects || _memo.Count > _options.MaxObjects)
            {
                throw new UnsafePickleException($"The pickle exceeds the limit of {_options.MaxObjects} objects.");
            }
        }
    }

    private object? ResolveGlobal(string module, string name)
    {
        object? resolved = (module, name) switch
        {
            ("torch._utils", "_rebuild_tensor_v2" or "_rebuild_tensor") => PickleGlobal.RebuildTensor,
            ("torch._utils", "_rebuild_parameter") => PickleGlobal.RebuildParameter,
            ("collections", "OrderedDict") => PickleGlobal.OrderedDict,
            ("torch", "Size") => PickleGlobal.TorchSize,
            ("_codecs", "encode") => PickleGlobal.CodecsEncode,
            ("torch", "DoubleStorage") => new PickleStorageType(TorchDType.Float64),
            ("torch", "FloatStorage") => new PickleStorageType(TorchDType.Float32),
            ("torch", "HalfStorage") => new PickleStorageType(TorchDType.Float16),
            ("torch", "BFloat16Storage") => new PickleStorageType(TorchDType.BFloat16),
            ("torch", "LongStorage") => new PickleStorageType(TorchDType.Int64),
            ("torch", "IntStorage") => new PickleStorageType(TorchDType.Int32),
            ("torch", "ShortStorage") => new PickleStorageType(TorchDType.Int16),
            ("torch", "CharStorage") => new PickleStorageType(TorchDType.Int8),
            ("torch", "ByteStorage") => new PickleStorageType(TorchDType.UInt8),
            ("torch", "BoolStorage") => new PickleStorageType(TorchDType.Bool),
            _ => null,
        };
        if (resolved is not null)
        {
            return resolved;
        }

        foreach (string prefix in _options.InertModulePrefixes)
        {
            if (module.StartsWith(prefix, StringComparison.Ordinal))
            {
                return PickleInert.Instance;
            }
        }

        throw Refused($"global '{module}.{name}'");
    }

    private static object? Call(object? callable, object?[] args)
    {
        switch (callable)
        {
            case PickleInert:
                return PickleInert.Instance;
            case PickleGlobal.OrderedDict:
                {
                    var dict = new PickleDict();
                    if (args.Length > 1)
                    {
                        throw Malformed("OrderedDict takes at most one argument");
                    }

                    if (args.Length == 1 && args[0] is List<object?> pairs)
                    {
                        foreach (object? pair in pairs)
                        {
                            if (pair is not object?[] { Length: 2 } kv || kv[0] is null)
                            {
                                throw Malformed("OrderedDict items must be (key, value) pairs");
                            }

                            dict.Set(kv[0]!, kv[1]);
                        }
                    }
                    else if (args.Length == 1)
                    {
                        throw Malformed("OrderedDict argument must be a list of pairs");
                    }

                    return dict;
                }

            case PickleGlobal.RebuildParameter:
                return args.Length >= 1 ? args[0] : throw Malformed("_rebuild_parameter without data");
            case PickleGlobal.TorchSize:
                return args is [object?[] dims] ? dims : throw Malformed("torch.Size takes one tuple");
            case PickleGlobal.CodecsEncode:
                return args is [string text, "latin1"] ? Encoding.Latin1.GetBytes(text) : throw Refused("_codecs.encode with an encoding other than latin1");
            case PickleGlobal.RebuildTensor:
                return RebuildTensor(args);
            default:
                throw Refused($"calling {callable?.GetType().Name ?? "None"}");
        }
    }

    private static PickleTensorRef RebuildTensor(object?[] args)
    {
        if (args.Length < 4 || args[0] is not PickleStorageRef storage || args[1] is not long offset
            || args[2] is not object?[] size || args[3] is not object?[] stride)
        {
            throw Malformed("_rebuild_tensor needs (storage, offset, size, stride)");
        }

        long[] shape = Dimensions(size);
        long[] strides = Dimensions(stride);
        if (shape.Length != strides.Length || offset < 0)
        {
            throw Malformed("tensor shape, stride and offset are inconsistent");
        }

        return new PickleTensorRef(storage, offset, shape, strides);
    }

    private static long[] Dimensions(object?[] values)
    {
        if (values.Length > MaxDims)
        {
            throw Malformed($"a tensor with more than {MaxDims} dimensions");
        }

        var result = new long[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] is not long value || value < 0)
            {
                throw Malformed("tensor dimensions must be non-negative integers");
            }

            result[i] = value;
        }

        return result;
    }

    private static PickleStorageRef ResolvePersistentId(object? pid)
    {
        if (pid is not object?[] { Length: >= 5 } parts || parts[0] is not "storage" || parts[1] is not PickleStorageType type
            || parts[2] is not string key || parts[4] is not long count || count < 0)
        {
            throw Refused("a persistent id other than ('storage', <storage type>, key, location, size)");
        }

        long elementOffset = 0;
        if (parts.Length >= 6 && parts[5] is object?[] { Length: 3 } view && view[1] is long viewOffset && viewOffset >= 0)
        {
            // Legacy storage view: (view key, offset, size) into the root storage.
            elementOffset = viewOffset;
        }

        return new PickleStorageRef(key, type.DType, count, elementOffset);
    }

    private static void AppendTo(object? target, List<object?> items)
    {
        switch (target)
        {
            case List<object?> list:
                list.AddRange(items);
                break;
            case PickleInert:
                break;
            default:
                throw Refused("APPEND on an object that is not a list");
        }
    }

    private static void SetItems(object? target, List<object?> flat)
    {
        if (target is PickleInert)
        {
            return;
        }

        if (target is not PickleDict dict || flat.Count % 2 != 0)
        {
            throw Refused("SETITEM on an object that is not a dictionary");
        }

        for (int i = 0; i < flat.Count; i += 2)
        {
            if (flat[i] is null or PickleDict or List<object?>)
            {
                throw Malformed("unhashable dictionary key");
            }

            dict.Set(flat[i]!, flat[i + 1]);
        }
    }

    private void Push(object? value) => _stack.Add(value);

    private object? Pop()
    {
        if (_stack.Count == 0)
        {
            throw Malformed("pop from an empty stack");
        }

        object? value = _stack[^1];
        _stack.RemoveAt(_stack.Count - 1);
        if (value is PickleMark)
        {
            throw Malformed("unexpected mark");
        }

        return value;
    }

    private object? PeekTop() => _stack.Count > 0 ? _stack[^1] : throw Malformed("empty stack");

    private int PopMark()
    {
        if (_marks.Count == 0)
        {
            throw Malformed("no mark on the stack");
        }

        int mark = _marks[^1];
        _marks.RemoveAt(_marks.Count - 1);
        return mark;
    }

    private void PushTuple(int count)
    {
        if (_stack.Count < count)
        {
            throw Malformed("tuple larger than the stack");
        }

        var items = _stack.GetRange(_stack.Count - count, count).ToArray();
        _stack.RemoveRange(_stack.Count - count, count);
        Push(items);
    }

    private void Memoize(long index)
    {
        _memo[index] = PeekTop();
    }

    private object? MemoGet(long index) => _memo.TryGetValue(index, out object? value) ? value : throw Malformed($"memo entry {index} does not exist");

    private byte ReadByte() => Take(1)[0];

    private int ReadLength32()
    {
        uint length = BinaryPrimitives.ReadUInt32LittleEndian(Take(4));
        return length <= int.MaxValue ? (int)length : throw Malformed("length too large");
    }

    private int ReadLength64()
    {
        ulong length = BinaryPrimitives.ReadUInt64LittleEndian(Take(8));
        return length <= int.MaxValue ? (int)length : throw Malformed("length too large");
    }

    private ReadOnlySpan<byte> Take(int count)
    {
        if (count < 0 || count > _memory.Length - _position)
        {
            throw Malformed("the stream is truncated");
        }

        var span = _memory.Span.Slice(_position, count);
        _position += count;
        return span;
    }

    private string ReadLine()
    {
        var rest = _memory.Span[_position..];
        int end = rest.IndexOf((byte)'\n');
        if (end < 0 || end > 4096)
        {
            throw Malformed("unterminated text line");
        }

        string line = Encoding.UTF8.GetString(rest[..end]);
        _position += end + 1;
        return line;
    }

    private object ReadLongBytes(int count)
    {
        var bytes = Take(count);
        if (bytes.Length == 0)
        {
            return 0L;
        }

        var value = new BigInteger(bytes, isUnsigned: false, isBigEndian: false);
        return value >= long.MinValue && value <= long.MaxValue ? (long)value : value;
    }

    private static object ParseTextInt(string text)
    {
        return text switch
        {
            "00" => false,
            "01" => true,
            _ => long.TryParse(text, out long value) ? value : throw Malformed("bad integer literal"),
        };
    }

    private static object ParseTextLong(string text)
    {
        string digits = text.EndsWith('L') ? text[..^1] : text;
        return long.TryParse(digits, out long value) ? value : throw Malformed("bad long literal");
    }

    private static UnsafePickleException Refused(string what) =>
        new($"The checkpoint's pickle uses {what}, which is not allowed in a tensor state dictionary. The file was not loaded and nothing in it was executed.");

    private static UnsafePickleException Malformed(string what) => new($"Malformed pickle: {what}.");
}
