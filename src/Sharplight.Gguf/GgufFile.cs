using System.Globalization;
using System.IO.MemoryMappedFiles;
using Sharplight.Tensors;

namespace Sharplight.Gguf;

/// <summary>
/// A memory-mapped, read-only GGUF file (versions 2 and 3). The mapping is created once for the whole
/// file; tensor data is never copied unless requested.
/// </summary>
/// <remarks>
/// Instances are immutable after <see cref="Open"/> and safe to read from several threads. Disposing unmaps the file:
/// weights and tensor pointers obtained from it must not be used afterwards (reads then throw
/// <see cref="ObjectDisposedException"/>).
/// </remarks>
public sealed unsafe class GgufFile : IDisposable
{
    /// <summary>"GGUF" as a little-endian <c>uint32</c>.</summary>
    internal const uint Magic = 0x46554747;

    /// <summary>Tensor data alignment when the file has no <c>general.alignment</c> key.</summary>
    internal const int DefaultAlignment = 32;

    private const int HeaderBytes = 24;
    private const int MaxTensorDimensions = 8;

    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _view;
    private byte* _base;
    private bool _pointerAcquired;
    private readonly Dictionary<string, object> _metadata;
    private readonly GgufTensorInfo[] _tensors;
    private readonly Dictionary<string, GgufTensorInfo> _byName;

    /// <summary>The path the file was opened from.</summary>
    public string Path { get; }

    /// <summary>The GGUF format version (2 or 3).</summary>
    public uint Version { get; }

    internal long FileLength { get; }

    /// <summary>Tensor data alignment in bytes (<c>general.alignment</c>, default 32).</summary>
    public int Alignment { get; }

    /// <summary>Absolute file offset of the tensor data section.</summary>
    public long DataOffset { get; }

    /// <summary>
    /// All key/value metadata. Values are CLR primitives (<c>byte</c>, <c>int</c>, <c>float</c>, <c>bool</c>, <c>string</c>, ...),
    /// typed arrays for GGUF arrays and <c>object[]</c> for arrays of arrays.
    /// </summary>
    public IReadOnlyDictionary<string, object> Metadata => _metadata;

    /// <summary>The tensors in file order.</summary>
    public IReadOnlyList<GgufTensorInfo> Tensors => _tensors;

    internal byte* BasePointer => _base;

    internal void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_base == null, this);

    private GgufFile(string path)
    {
        Path = path;
        FileLength = new FileInfo(path).Length;
        if (FileLength < HeaderBytes)
        {
            throw new InvalidDataException("File too small to be GGUF.");
        }

        _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        try
        {
            _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            _pointerAcquired = true;
            _base = pointer + _view.PointerOffset;

            var reader = new GgufReader(_base, FileLength);
            Version = ReadVersion(ref reader);
            long tensorCount = checked((long)reader.ReadU64());
            long kvCount = checked((long)reader.ReadU64());
            if (tensorCount < 0 || kvCount < 0 || tensorCount > FileLength || kvCount > FileLength)
            {
                throw new InvalidDataException("Implausible tensor/KV count.");
            }

            _metadata = ReadMetadata(ref reader, kvCount);
            Alignment = ReadAlignment(_metadata);
            var rawTensors = ReadTensorHeaders(ref reader, tensorCount);
            DataOffset = (reader.Position + Alignment - 1) / Alignment * Alignment;

            _tensors = new GgufTensorInfo[tensorCount];
            _byName = new Dictionary<string, GgufTensorInfo>((int)tensorCount, StringComparer.Ordinal);
            BuildTensorTable(rawTensors);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>Opens and memory-maps <paramref name="path"/>, parsing the header, metadata and tensor table.</summary>
    /// <exception cref="InvalidDataException">The file is not a well-formed GGUF file.</exception>
    /// <exception cref="NotSupportedException">The GGUF version or a tensor type is not supported.</exception>
    public static GgufFile Open(string path) => new(path);

    private static uint ReadVersion(ref GgufReader reader)
    {
        if (reader.ReadU32() != Magic)
        {
            throw new InvalidDataException("Not a GGUF file (bad magic).");
        }

        uint version = reader.ReadU32();
        if (version is not (2 or 3))
        {
            throw new NotSupportedException($"Unsupported GGUF version {version}.");
        }

        return version;
    }

    private static Dictionary<string, object> ReadMetadata(ref GgufReader reader, long kvCount)
    {
        var metadata = new Dictionary<string, object>((int)Math.Min(kvCount, 1 << 20), StringComparer.Ordinal);
        for (long i = 0; i < kvCount; i++)
        {
            string key = reader.ReadString();
            var type = (GgufValueType)reader.ReadU32();
            metadata[key] = reader.ReadValue(type);
        }

        return metadata;
    }

    private static int ReadAlignment(Dictionary<string, object> metadata)
    {
        if (metadata.TryGetValue("general.alignment", out var value))
        {
            int alignment = Convert.ToInt32(value, CultureInfo.InvariantCulture);
            if (alignment > 0)
            {
                return alignment;
            }
        }

        return DefaultAlignment;
    }

    private static (string Name, int Type, long[] Ne, long Offset)[] ReadTensorHeaders(ref GgufReader reader, long tensorCount)
    {
        var headers = new (string Name, int Type, long[] Ne, long Offset)[tensorCount];
        for (long i = 0; i < tensorCount; i++)
        {
            string name = reader.ReadString();
            uint dimensionCount = reader.ReadU32();
            if (dimensionCount > MaxTensorDimensions)
            {
                throw new InvalidDataException($"Tensor '{name}' has {dimensionCount} dimensions.");
            }

            var ne = new long[dimensionCount];
            for (int d = 0; d < dimensionCount; d++)
            {
                ne[d] = checked((long)reader.ReadU64());
            }

            int type = (int)reader.ReadU32();
            long offset = checked((long)reader.ReadU64());
            headers[i] = (name, type, ne, offset);
        }

        return headers;
    }

    private void BuildTensorTable((string Name, int Type, long[] Ne, long Offset)[] headers)
    {
        for (int i = 0; i < headers.Length; i++)
        {
            var (name, type, ne, offset) = headers[i];
            var dataType = GgmlTypes.GetDataType(type);
            long size = ComputeByteSize(dataType, ne, name);
            long absoluteOffset = DataOffset + offset;
            if (absoluteOffset < 0 || absoluteOffset + size > FileLength)
            {
                throw new InvalidDataException($"Tensor '{name}' extends beyond the end of the file (offset {absoluteOffset}, size {size}, file {FileLength}).");
            }

            var info = new GgufTensorInfo(this, name, type, dataType, ne, offset, absoluteOffset, size);
            _tensors[i] = info;
            _byName[name] = info;
        }
    }

    internal static long ComputeByteSize(DataType dt, long[] ne, string name)
    {
        long elements = 1;
        foreach (var n in ne)
        {
            elements = checked(elements * n);
        }

        long rowLength = ne.Length > 0 ? ne[0] : 1;
        if (rowLength % dt.BlockSize != 0)
        {
            throw new InvalidDataException($"Tensor '{name}': row length {rowLength} is not a multiple of {dt} block size {dt.BlockSize}.");
        }

        return checked(elements / dt.BlockSize * dt.BlockBytes);
    }

    /// <summary>The <c>general.architecture</c> value, or null when the file has none.</summary>
    public string? Architecture => TryGet<string>("general.architecture", out var architecture) ? architecture : null;

    internal GgufTensorInfo? FindTensor(string name) => _byName.GetValueOrDefault(name);

    internal GgufTensorInfo GetTensor(string name)
        => _byName.TryGetValue(name, out var tensor) ? tensor : throw new KeyNotFoundException($"Tensor '{name}' not found.");

    // ---- typed metadata helpers -----------------------------------------------------------

    /// <summary>Reads metadata <paramref name="key"/> as <typeparamref name="T"/>; numeric values convert, strings and other types do not.</summary>
    internal bool TryGet<T>(string key, out T value)
    {
        if (_metadata.TryGetValue(key, out var raw) && GgufConvert.TryConvert(raw, out T? converted))
        {
            value = converted!;
            return true;
        }

        value = default!;
        return false;
    }

    internal T Get<T>(string key) => TryGet<T>(key, out var v) ? v
        : throw new KeyNotFoundException($"Metadata key '{key}' missing or not convertible to {typeof(T).Name}.");

    internal string GetString(string key) => Get<string>(key);

    internal uint GetUInt32(string key) => Get<uint>(key);

    internal uint GetUInt32(string key, uint defaultValue) => TryGet<uint>(key, out var v) ? v : defaultValue;

    internal long GetInt64(string key) => Get<long>(key);

    internal float GetFloat(string key) => Get<float>(key);

    internal T[] GetArray<T>(string key) => Get<T[]>(key);

    /// <summary>Reads <c>{general.architecture}.{suffix}</c>.</summary>
    internal T GetArch<T>(string suffix)
        => Get<T>($"{Architecture ?? throw new InvalidOperationException("general.architecture missing")}.{suffix}");

    /// <summary>Unmaps the file. Safe to call more than once.</summary>
    public void Dispose()
    {
        if (_pointerAcquired)
        {
            _view!.SafeMemoryMappedViewHandle.ReleasePointer();
            _pointerAcquired = false;
        }

        _base = null;
        _view?.Dispose();
        _view = null;
        _mmf?.Dispose();
        _mmf = null;
    }
}
