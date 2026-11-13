using System.Runtime.CompilerServices;
using System.IO.MemoryMappedFiles;
using System.Text;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

public enum GgufValueType : uint
{
    UInt8 = 0, Int8 = 1, UInt16 = 2, Int16 = 3, UInt32 = 4, Int32 = 5, Float32 = 6, Bool = 7,
    String = 8, Array = 9, UInt64 = 10, Int64 = 11, Float64 = 12,
}

/// <summary>
/// A memory-mapped, read-only GGUF file (versions 2 and 3). The mapping is created once for the whole
/// file; tensor data is never copied unless requested.
/// </summary>
public sealed unsafe class GgufFile : IDisposable
{
    public const uint Magic = 0x46554747; // "GGUF" little-endian
    public const int DefaultAlignment = 32;

    MemoryMappedFile? _mmf;
    MemoryMappedViewAccessor? _view;
    byte* _base;
    bool _pointerAcquired;
    readonly Dictionary<string, object> _metadata;
    readonly GgufTensorInfo[] _tensors;
    readonly Dictionary<string, GgufTensorInfo> _byName;

    public string Path { get; }
    public uint Version { get; }
    public long FileLength { get; }
    public int Alignment { get; }
    /// <summary>Absolute file offset of the tensor data section.</summary>
    public long DataOffset { get; }
    public IReadOnlyDictionary<string, object> Metadata => _metadata;
    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;

    internal byte* BasePointer => _base;
    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_base == null, this);

    GgufFile(string path)
    {
        Path = path;
        var fi = new FileInfo(path);
        FileLength = fi.Length;
        if (FileLength < 24) throw new InvalidDataException("File too small to be GGUF.");

        _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        try
        {
            _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* p = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            _pointerAcquired = true;
            _base = p + _view.PointerOffset;

            var r = new Reader(_base, FileLength);
            if (r.ReadU32() != Magic) throw new InvalidDataException("Not a GGUF file (bad magic).");
            Version = r.ReadU32();
            if (Version is not (2 or 3)) throw new NotSupportedException($"Unsupported GGUF version {Version}.");
            long tensorCount = checked((long)r.ReadU64());
            long kvCount = checked((long)r.ReadU64());
            if (tensorCount < 0 || kvCount < 0 || tensorCount > FileLength || kvCount > FileLength)
                throw new InvalidDataException("Implausible tensor/KV count.");

            _metadata = new Dictionary<string, object>((int)Math.Min(kvCount, 1 << 20), StringComparer.Ordinal);
            for (long i = 0; i < kvCount; i++)
            {
                string key = r.ReadString();
                var type = (GgufValueType)r.ReadU32();
                _metadata[key] = r.ReadValue(type);
            }

            Alignment = DefaultAlignment;
            if (_metadata.TryGetValue("general.alignment", out var al))
            {
                int a = Convert.ToInt32(al, System.Globalization.CultureInfo.InvariantCulture);
                if (a > 0) Alignment = a;
            }

            var raw = new (string name, int type, long[] ne, long off)[tensorCount];
            for (long i = 0; i < tensorCount; i++)
            {
                string name = r.ReadString();
                uint nd = r.ReadU32();
                if (nd > 8) throw new InvalidDataException($"Tensor '{name}' has {nd} dimensions.");
                var ne = new long[nd];
                for (int d = 0; d < nd; d++) ne[d] = checked((long)r.ReadU64());
                int type = (int)r.ReadU32();
                long off = checked((long)r.ReadU64());
                raw[i] = (name, type, ne, off);
            }

            DataOffset = (r.Position + Alignment - 1) / Alignment * Alignment;

            _tensors = new GgufTensorInfo[tensorCount];
            _byName = new Dictionary<string, GgufTensorInfo>((int)tensorCount, StringComparer.Ordinal);
            for (int i = 0; i < raw.Length; i++)
            {
                var (name, type, ne, off) = raw[i];
                var dt = GgmlTypes.GetDataType(type);
                long size = ComputeByteSize(dt, ne, name);
                long abs = DataOffset + off;
                if (abs < 0 || abs + size > FileLength)
                    throw new InvalidDataException($"Tensor '{name}' extends beyond the end of the file (offset {abs}, size {size}, file {FileLength}).");
                var info = new GgufTensorInfo(this, name, type, dt, ne, off, abs, size);
                _tensors[i] = info;
                _byName[name] = info;
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    internal static long ComputeByteSize(DataType dt, long[] ne, string name)
    {
        long elems = 1;
        foreach (var n in ne) elems = checked(elems * n);
        if (ne.Length == 0) elems = 1;
        long row = ne.Length > 0 ? ne[0] : 1;
        if (row % dt.BlockSize != 0)
            throw new InvalidDataException($"Tensor '{name}': row length {row} is not a multiple of {dt} block size {dt.BlockSize}.");
        return checked(elems / dt.BlockSize * dt.BlockBytes);
    }

    public static GgufFile Open(string path) => new(path);

    public string? Architecture => TryGet<string>("general.architecture", out var a) ? a : null;

    public GgufTensorInfo? FindTensor(string name) => _byName.GetValueOrDefault(name);
    public bool TryGetTensor(string name, out GgufTensorInfo info) => _byName.TryGetValue(name, out info!);
    public GgufTensorInfo GetTensor(string name) => _byName.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"Tensor '{name}' not found.");

    // ---- typed metadata helpers -----------------------------------------------------------
    public bool ContainsKey(string key) => _metadata.ContainsKey(key);

    public bool TryGet<T>(string key, out T value)
    {
        if (_metadata.TryGetValue(key, out var o) && GgufConvert.TryConvert(o, out T? v)) { value = v!; return true; }
        value = default!;
        return false;
    }

    public T Get<T>(string key) => TryGet<T>(key, out var v) ? v
        : throw new KeyNotFoundException($"Metadata key '{key}' missing or not convertible to {typeof(T).Name}.");

    public string GetString(string key) => Get<string>(key);
    public string GetString(string key, string defaultValue) => TryGet<string>(key, out var v) ? v : defaultValue;
    public uint GetUInt32(string key) => Get<uint>(key);
    public uint GetUInt32(string key, uint defaultValue) => TryGet<uint>(key, out var v) ? v : defaultValue;
    public long GetInt64(string key) => Get<long>(key);
    public long GetInt64(string key, long defaultValue) => TryGet<long>(key, out var v) ? v : defaultValue;
    public float GetFloat(string key) => Get<float>(key);
    public float GetFloat(string key, float defaultValue) => TryGet<float>(key, out var v) ? v : defaultValue;
    public bool GetBool(string key) => Get<bool>(key);
    public bool GetBool(string key, bool defaultValue) => TryGet<bool>(key, out var v) ? v : defaultValue;
    public T[] GetArray<T>(string key) => Get<T[]>(key);

    string ArchKey(string suffix) => $"{Architecture ?? throw new InvalidOperationException("general.architecture missing")}.{suffix}";
    /// <summary>Reads <c>{general.architecture}.{suffix}</c>.</summary>
    public T GetArch<T>(string suffix) => Get<T>(ArchKey(suffix));
    public T GetArch<T>(string suffix, T defaultValue) => TryGet<T>(ArchKey(suffix), out var v) ? v : defaultValue;
    public bool TryGetArch<T>(string suffix, out T value)
    {
        if (Architecture is null) { value = default!; return false; }
        return TryGet(ArchKey(suffix), out value);
    }

    public void Dispose()
    {
        if (_pointerAcquired) { _view!.SafeMemoryMappedViewHandle.ReleasePointer(); _pointerAcquired = false; }
        _base = null;
        _view?.Dispose(); _view = null;
        _mmf?.Dispose(); _mmf = null;
    }

    // ---- parsing -------------------------------------------------------------------------
    ref struct Reader
    {
        readonly byte* _p;
        readonly long _len;
        long _pos;
        public Reader(byte* p, long len) { _p = p; _len = len; _pos = 0; }
        public readonly long Position => _pos;

        byte* Take(long n)
        {
            if (n < 0 || _pos + n > _len) throw new InvalidDataException("Unexpected end of GGUF file.");
            var r = _p + _pos; _pos += n; return r;
        }
        public uint ReadU32() => Unsafe.ReadUnaligned<uint>(Take(4));
        public ulong ReadU64() => Unsafe.ReadUnaligned<ulong>(Take(8));

        public string ReadString()
        {
            long n = checked((long)ReadU64());
            if (n > int.MaxValue) throw new InvalidDataException("String too long.");
            if (n == 0) return "";
            return Encoding.UTF8.GetString(Take(n), (int)n);
        }

        public object ReadValue(GgufValueType t)
        {
            switch (t)
            {
                case GgufValueType.UInt8: return *Take(1);
                case GgufValueType.Int8: return (sbyte)*Take(1);
                case GgufValueType.UInt16: return Unsafe.ReadUnaligned<ushort>(Take(2));
                case GgufValueType.Int16: return Unsafe.ReadUnaligned<short>(Take(2));
                case GgufValueType.UInt32: return Unsafe.ReadUnaligned<uint>(Take(4));
                case GgufValueType.Int32: return Unsafe.ReadUnaligned<int>(Take(4));
                case GgufValueType.Float32: return Unsafe.ReadUnaligned<float>(Take(4));
                case GgufValueType.Bool: return *Take(1) != 0;
                case GgufValueType.String: return ReadString();
                case GgufValueType.UInt64: return Unsafe.ReadUnaligned<ulong>(Take(8));
                case GgufValueType.Int64: return Unsafe.ReadUnaligned<long>(Take(8));
                case GgufValueType.Float64: return Unsafe.ReadUnaligned<double>(Take(8));
                case GgufValueType.Array: return ReadArray();
                default: throw new InvalidDataException($"Unknown GGUF value type {(uint)t}.");
            }
        }

        T[] ReadBlock<T>(long n) where T : unmanaged
        {
            var arr = new T[n];
            if (n > 0) new ReadOnlySpan<byte>(Take(n * sizeof(T)), checked((int)(n * sizeof(T)))).CopyTo(System.Runtime.InteropServices.MemoryMarshal.AsBytes(arr.AsSpan()));
            return arr;
        }

        object ReadArray()
        {
            var et = (GgufValueType)ReadU32();
            long n = checked((long)ReadU64());
            if (n < 0 || n > int.MaxValue) throw new InvalidDataException("Array too large.");
            switch (et)
            {
                case GgufValueType.UInt8: return ReadBlock<byte>(n);
                case GgufValueType.Int8: return ReadBlock<sbyte>(n);
                case GgufValueType.UInt16: return ReadBlock<ushort>(n);
                case GgufValueType.Int16: return ReadBlock<short>(n);
                case GgufValueType.UInt32: return ReadBlock<uint>(n);
                case GgufValueType.Int32: return ReadBlock<int>(n);
                case GgufValueType.Float32: return ReadBlock<float>(n);
                case GgufValueType.UInt64: return ReadBlock<ulong>(n);
                case GgufValueType.Int64: return ReadBlock<long>(n);
                case GgufValueType.Float64: return ReadBlock<double>(n);
                case GgufValueType.Bool:
                {
                    var b = new bool[n];
                    var src = new ReadOnlySpan<byte>(Take(n), (int)n);
                    for (int i = 0; i < b.Length; i++) b[i] = src[i] != 0;
                    return b;
                }
                case GgufValueType.String:
                {
                    // each string needs at least 8 bytes; guards against absurd counts
                    if (n * 8 > _len - _pos) throw new InvalidDataException("String array larger than file.");
                    var s = new string[n];
                    for (int i = 0; i < s.Length; i++) s[i] = ReadString();
                    return s;
                }
                case GgufValueType.Array:
                {
                    if (n * 12 > _len - _pos) throw new InvalidDataException("Array larger than file.");
                    var o = new object[n];
                    for (int i = 0; i < o.Length; i++) o[i] = ReadArray();
                    return o;
                }
                default: throw new InvalidDataException($"Unknown GGUF array element type {(uint)et}.");
            }
        }
    }
}

static class GgufConvert
{
    public static bool TryConvert<T>(object o, out T? value)
    {
        if (o is T t) { value = t; return true; }
        var target = typeof(T);
        try
        {
            if (target.IsArray)
            {
                var et = target.GetElementType()!;
                if (o is Array src)
                {
                    var dst = Array.CreateInstance(et, src.Length);
                    for (int i = 0; i < src.Length; i++)
                    {
                        var v = src.GetValue(i)!;
                        if (!TryConvertTo(et, v, out var c)) { value = default; return false; }
                        dst.SetValue(c, i);
                    }
                    value = (T)(object)dst; return true;
                }
                value = default; return false;
            }
            if (TryConvertTo(target, o, out var r)) { value = (T)r!; return true; }
        }
        catch (Exception e) when (e is InvalidCastException or OverflowException or FormatException) { }
        value = default; return false;
    }

    static bool TryConvertTo(Type target, object v, out object? result)
    {
        result = null;
        if (target.IsInstanceOfType(v)) { result = v; return true; }
        if (target == typeof(object)) { result = v; return true; }
        if (v is string || target == typeof(string) || v is Array) return false;
        if (target.IsPrimitive || target == typeof(decimal))
        {
            try { result = Convert.ChangeType(v, target, System.Globalization.CultureInfo.InvariantCulture); return true; }
            catch (Exception e) when (e is InvalidCastException or OverflowException) { return false; }
        }
        return false;
    }
}
