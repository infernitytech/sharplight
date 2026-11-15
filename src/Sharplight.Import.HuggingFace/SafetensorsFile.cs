using System.IO.MemoryMappedFiles;
using System.Text.Json;
using Sharplight.Import.PyTorch;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// A memory-mapped, read-only safetensors file. Only the JSON header is parsed; tensor data is never
/// copied unless requested. Disposing unmaps the file: weights obtained from it must not be read afterwards.
/// </summary>
internal sealed unsafe class SafetensorsFile : IDisposable
{
    private const long MaxHeaderBytes = 256L << 20;
    private const int HeaderLengthBytes = sizeof(ulong);

    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _view;
    private byte* _base;
    private TorchCheckpointFile? _torch;
    private readonly Dictionary<string, SafetensorsTensor> _tensors = new(StringComparer.Ordinal);

    /// <summary>The path the file was opened from.</summary>
    public string Path { get; }

    /// <summary>Absolute file offset of the data section.</summary>
    public long DataOffset { get; }

    /// <summary>The tensors of this file by name.</summary>
    public IReadOnlyDictionary<string, SafetensorsTensor> Tensors => _tensors;

    /// <summary>The start of the mapping; throws <see cref="ObjectDisposedException"/> after disposal.</summary>
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
        long fileLength = new FileInfo(path).Length;
        if (fileLength < HeaderLengthBytes)
        {
            throw new InvalidDataException($"{path}: too small to be a safetensors file.");
        }

        _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        try
        {
            _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            _base = pointer + _view.PointerOffset;

            ulong headerLength = *(ulong*)_base; // little-endian on all supported platforms
            if (headerLength > (ulong)MaxHeaderBytes || HeaderLengthBytes + (long)headerLength > fileLength)
            {
                throw new InvalidDataException($"{path}: invalid safetensors header length {headerLength}.");
            }

            DataOffset = HeaderLengthBytes + (long)headerLength;
            long dataLength = fileLength - DataOffset;

            var header = new ReadOnlySpan<byte>(_base + HeaderLengthBytes, (int)headerLength).ToArray();
            using var doc = JsonDocument.Parse(header);
            foreach (var property in doc.RootElement.EnumerateObject())
            {
                if (property.Name != "__metadata__")
                {
                    _tensors[property.Name] = ParseTensor(property, dataLength);
                }
            }
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    private SafetensorsFile(TorchCheckpointFile torch)
    {
        _torch = torch;
        Path = torch.Path;
        DataOffset = 0;
        _base = torch.BasePointer;
        foreach (var (name, entry) in torch.Tensors)
        {
            string dtype = TorchDTypeName(entry.DType);
            _tensors[name] = new SafetensorsTensor
            {
                Name = name,
                DType = dtype,
                Type = SafetensorsTensor.MapDType(dtype),
                Shape = entry.Shape,
                Offset = entry.FileOffset,
                ByteLength = entry.ByteLength,
                File = this,
            };
        }
    }

    /// <summary>
    /// Opens a PyTorch checkpoint (<c>.bin</c> / <c>.pt</c> / <c>.pth</c> / <c>.ckpt</c>) through the restricted unpickler of
    /// <c>Sharplight.Import.PyTorch</c> and presents its tensors like safetensors entries (offsets are absolute file offsets).
    /// </summary>
    /// <param name="path">The checkpoint file.</param>
    /// <param name="options">Unpickler settings, or <see langword="null"/> for <see cref="TorchCheckpointOptions.Default"/>.</param>
    /// <returns>The opened file.</returns>
    /// <exception cref="UnsafePickleException">The pickle uses an opcode or global outside the tensor state-dict subset.</exception>
    public static SafetensorsFile OpenTorch(string path, TorchCheckpointOptions? options = null) => new(TorchCheckpointFile.Open(path, options));

    private static string TorchDTypeName(TorchDType dtype) => dtype switch
    {
        TorchDType.Float64 => "F64",
        TorchDType.Float32 => "F32",
        TorchDType.Float16 => "F16",
        TorchDType.BFloat16 => "BF16",
        TorchDType.Int64 => "I64",
        TorchDType.Int32 => "I32",
        TorchDType.Int16 => "I16",
        TorchDType.Int8 => "I8",
        TorchDType.UInt8 => "U8",
        _ => "BOOL",
    };

    /// <summary>Opens and memory-maps <paramref name="path"/> and parses its header.</summary>
    /// <exception cref="InvalidDataException">The header is malformed or a tensor lies outside the data section.</exception>
    public static SafetensorsFile Open(string path) => new(path);

    private SafetensorsTensor ParseTensor(JsonProperty property, long dataLength)
    {
        string name = property.Name;
        var entry = property.Value;
        string dtype = entry.GetProperty("dtype").GetString() ?? throw new InvalidDataException($"{Path}: tensor '{name}' has no dtype.");
        var shape = entry.GetProperty("shape").EnumerateArray().Select(x => x.GetInt64()).ToArray();
        var offsets = entry.GetProperty("data_offsets");
        long begin = offsets[0].GetInt64();
        long end = offsets[1].GetInt64();
        if (begin < 0 || end < begin || end > dataLength)
        {
            throw new InvalidDataException($"{Path}: tensor '{name}' data offsets [{begin},{end}) are outside the data section ({dataLength} bytes).");
        }

        int elementBytes = SafetensorsTensor.ElementBytes(dtype);
        long elements = 1;
        foreach (var dim in shape)
        {
            elements = checked(elements * dim);
        }

        if (elementBytes != 0 && elements * elementBytes != end - begin)
        {
            throw new InvalidDataException($"{Path}: tensor '{name}' ({dtype} {string.Join('x', shape)}) has {end - begin} bytes, expected {elements * elementBytes}.");
        }

        return new SafetensorsTensor
        {
            Name = name,
            DType = dtype,
            Type = SafetensorsTensor.MapDType(dtype),
            Shape = shape,
            Offset = begin,
            ByteLength = end - begin,
            File = this,
        };
    }

    /// <summary>Unmaps the file. Safe to call more than once.</summary>
    public void Dispose()
    {
        if (_torch != null)
        {
            _torch.Dispose();
            _torch = null;
            _base = null;
            return;
        }

        if (_base != null)
        {
            _view?.SafeMemoryMappedViewHandle.ReleasePointer();
        }

        _base = null;
        _view?.Dispose();
        _view = null;
        _mmf?.Dispose();
        _mmf = null;
    }
}
