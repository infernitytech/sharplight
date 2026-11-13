using System.IO.MemoryMappedFiles;
using System.Text.Json;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>One tensor entry of a safetensors header.</summary>
public sealed class SafetensorsTensor
{
    public required string Name { get; init; }
    /// <summary>The safetensors dtype string, e.g. "BF16".</summary>
    public required string DType { get; init; }
    /// <summary>Sharplight type, or null if the dtype has no Sharplight equivalent (reading it throws).</summary>
    public DataType? Type { get; init; }
    /// <summary>Shape, outermost dimension first (same convention as <see cref="Shape"/>).</summary>
    public required long[] Shape { get; init; }
    /// <summary>Offset relative to the start of the data section.</summary>
    public required long Offset { get; init; }
    public required long ByteLength { get; init; }
    public required SafetensorsFile File { get; init; }

    public long Elements => Shape.Aggregate(1L, (a, b) => a * b);

    /// <summary>A zero-copy weight view over the memory map.</summary>
    public WeightInfo ToWeightInfo()
    {
        var type = Type ?? throw new NotSupportedException(
            $"{File.Path}: tensor '{Name}' has unsupported safetensors dtype '{DType}'.");
        return new WeightInfo
        {
            Name = Name,
            Type = type,
            Shape = new Shape(Shape.Select(d => (Dim)d)),
            Source = new SafetensorsWeightSource(File, File.DataOffset + Offset, ByteLength),
        };
    }

    internal static DataType? MapDType(string dtype) => dtype switch
    {
        "F32" => DataType.F32,
        "F16" => DataType.F16,
        "BF16" => DataType.BF16,
        "F64" => DataType.F64,
        "I8" => DataType.I8,
        "U8" => DataType.U8,
        "I16" => DataType.I16,
        "I32" => DataType.I32,
        "U32" => DataType.U32,
        "I64" => DataType.I64,
        "BOOL" => DataType.Bool,
        _ => null, // F8_*, U16, U64, F4, ... : parsed but not readable
    };

    internal static int ElementBytes(string dtype) => dtype switch
    {
        "F64" or "I64" or "U64" => 8,
        "F32" or "I32" or "U32" => 4,
        "F16" or "BF16" or "I16" or "U16" => 2,
        "I8" or "U8" or "BOOL" or "F8_E4M3" or "F8_E5M2" or "F8_E8M0" => 1,
        _ => 0, // unknown / sub-byte
    };
}

/// <summary>Zero-copy view of a byte range inside a memory-mapped safetensors file.</summary>
public sealed unsafe class SafetensorsWeightSource(SafetensorsFile file, long fileOffset, long length) : IWeightSource
{
    public long ByteLength => length;

    public void Read(long offset, Span<byte> destination)
    {
        if (offset < 0 || offset + destination.Length > length)
        {
            throw new ArgumentOutOfRangeException(nameof(offset));
        }

        new ReadOnlySpan<byte>(file.BasePointer + fileOffset + offset, destination.Length).CopyTo(destination);
    }

    public nint TryGetPointer() => (nint)(file.BasePointer + fileOffset);
}

/// <summary>
/// A memory-mapped, read-only safetensors file. Only the JSON header is parsed; tensor data is never
/// copied unless requested.
/// </summary>
public sealed unsafe class SafetensorsFile : IDisposable
{
    private const long MaxHeaderBytes = 256L << 20;

    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _view;
    private byte* _base;
    private readonly Dictionary<string, SafetensorsTensor> _tensors = new(StringComparer.Ordinal);

    public string Path { get; }
    public long FileLength { get; }
    /// <summary>Absolute file offset of the data section.</summary>
    public long DataOffset { get; }
    public IReadOnlyDictionary<string, SafetensorsTensor> Tensors => _tensors;
    /// <summary>The optional <c>__metadata__</c> string map.</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    internal byte* BasePointer
    {
        get
        {
            ObjectDisposedException.ThrowIf(_base == null, this);
            return _base;
        }
    }

    private SafetensorsFile(string path)
    {
        Path = path;
        FileLength = new FileInfo(path).Length;
        if (FileLength < 8)
        {
            throw new InvalidDataException($"{path}: too small to be a safetensors file.");
        }

        _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        try
        {
            _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* p = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
            _base = p + _view.PointerOffset;

            ulong headerLen = *(ulong*)_base; // little-endian on all supported platforms
            if (headerLen > (ulong)MaxHeaderBytes || 8 + (long)headerLen > FileLength)
            {
                throw new InvalidDataException($"{path}: invalid safetensors header length {headerLen}.");
            }

            DataOffset = 8 + (long)headerLen;
            long dataLen = FileLength - DataOffset;

            var meta = new Dictionary<string, string>(StringComparer.Ordinal);
            using var doc = JsonDocument.Parse(new ReadOnlyMemory<byte>(UnmanagedToArray(_base + 8, (int)headerLen)));
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.Name == "__metadata__")
                {
                    foreach (var m in prop.Value.EnumerateObject())
                    {
                        meta[m.Name] = m.Value.GetString() ?? "";
                    }

                    continue;
                }
                var e = prop.Value;
                string dtype = e.GetProperty("dtype").GetString() ?? throw new InvalidDataException($"{path}: tensor '{prop.Name}' has no dtype.");
                var shape = e.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
                var offs = e.GetProperty("data_offsets");
                long begin = offs[0].GetInt64(), end = offs[1].GetInt64();
                if (begin < 0 || end < begin || end > dataLen)
                {
                    throw new InvalidDataException($"{path}: tensor '{prop.Name}' data offsets [{begin},{end}) are outside the data section ({dataLen} bytes).");
                }

                int eb = SafetensorsTensor.ElementBytes(dtype);
                long elems = 1;
                foreach (var d in shape)
                {
                    elems = checked(elems * d);
                }

                if (eb != 0 && elems * eb != end - begin)
                {
                    throw new InvalidDataException($"{path}: tensor '{prop.Name}' ({dtype} {string.Join('x', shape)}) has {end - begin} bytes, expected {elems * eb}.");
                }

                _tensors[prop.Name] = new SafetensorsTensor
                {
                    Name = prop.Name, DType = dtype, Type = SafetensorsTensor.MapDType(dtype),
                    Shape = shape, Offset = begin, ByteLength = end - begin, File = this,
                };
            }
            Metadata = meta;
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private static byte[] UnmanagedToArray(byte* p, int len) => new ReadOnlySpan<byte>(p, len).ToArray();

    public static SafetensorsFile Open(string path) => new(path);

    public void Dispose()
    {
        if (_base != null)
        {
            _view?.SafeMemoryMappedViewHandle.ReleasePointer();
        }

        _base = null;
        _view?.Dispose(); _view = null;
        _mmf?.Dispose(); _mmf = null;
    }
}

/// <summary>
/// All shards of a safetensors checkpoint. Shards are discovered via <c>model.safetensors.index.json</c>
/// (<c>weight_map</c>) or, without an index, as all <c>*.safetensors</c> files in the directory.
/// </summary>
public sealed class SafetensorsCheckpoint : IDisposable
{
    private readonly List<SafetensorsFile> _files = [];
    private readonly Dictionary<string, SafetensorsTensor> _tensors = new(StringComparer.Ordinal);

    public string Directory { get; }
    public IReadOnlyList<SafetensorsFile> Files => _files;
    public IReadOnlyDictionary<string, SafetensorsTensor> Tensors => _tensors;

    private SafetensorsCheckpoint(string dir) => Directory = dir;

    public static SafetensorsCheckpoint Open(string directory)
    {
        var cp = new SafetensorsCheckpoint(directory);
        try
        {
            foreach (var shard in DiscoverShards(directory))
            {
                var f = SafetensorsFile.Open(shard);
                cp._files.Add(f);
                foreach (var (name, t) in f.Tensors)
                {
                    if (!cp._tensors.TryAdd(name, t))
                    {
                        throw new InvalidDataException($"Tensor '{name}' appears in more than one shard ({cp._tensors[name].File.Path}, {shard}).");
                    }
                }
            }
            if (cp._files.Count == 0)
            {
                throw new FileNotFoundException($"No *.safetensors files in '{directory}'.");
            }

            return cp;
        }
        catch
        {
            cp.Dispose();
            throw;
        }
    }

    private static IEnumerable<string> DiscoverShards(string dir)
    {
        var all = System.IO.Directory.EnumerateFiles(dir, "*.safetensors").OrderBy(p => p, StringComparer.Ordinal).ToList();
        string? index = System.IO.Directory.EnumerateFiles(dir, "*.safetensors.index.json").OrderBy(p => p, StringComparer.Ordinal).FirstOrDefault();
        if (index == null)
        {
            return all;
        }

        using var doc = JsonDocument.Parse(File.ReadAllBytes(index));
        var names = new SortedSet<string>(StringComparer.Ordinal);
        if (doc.RootElement.TryGetProperty("weight_map", out var wm))
        {
            foreach (var p in wm.EnumerateObject())
            {
                if (p.Value.GetString() is { } s)
                {
                    names.Add(s);
                }
            }
        }

        if (names.Count == 0)
        {
            return all;
        }

        return names.Select(n =>
        {
            var full = System.IO.Path.Combine(dir, n);
            return File.Exists(full) ? full : throw new FileNotFoundException($"Shard '{n}' listed in {System.IO.Path.GetFileName(index)} is missing.", full);
        });
    }

    public void Dispose()
    {
        foreach (var f in _files)
        {
            f.Dispose();
        }

        _files.Clear();
    }
}
