using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using Sharplight.Tensors;

namespace Sharplight.Import.WhisperCpp;

/// <summary>
/// A memory-mapped whisper.cpp model file (the legacy ggml container written by whisper.cpp's <c>convert-pt-to-ggml.py</c> and
/// <c>quantize</c>): the magic <c>ggml</c>, eleven int32 hyper-parameters, the mel filter bank, the vocabulary (byte strings) and the
/// tensors (dimension count, name length, ggml type, ggml dimensions innermost first, name, unaligned data) up to the end of the file.
/// </summary>
internal sealed unsafe class WhisperCppFile : IDisposable
{
    /// <summary>The magic number <c>ggml</c> as a little-endian uint32.</summary>
    public const uint Magic = 0x67676d6c;

    private const int FileTypeQuantizationFactor = 1000;

    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private readonly long _length;
    private byte* _base;

    private WhisperCppFile(string path)
    {
        // The stream's length is the target's when the path is a symbolic link (FileInfo reports the link's own size).
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        _length = stream.Length;
        _file = MemoryMappedFile.CreateFromFile(stream, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
        try
        {
            _view = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            _base = pointer + _view.PointerOffset;
            long position = 0;
            if (ReadUInt32(ref position) != Magic)
            {
                throw new InvalidDataException($"{path} is not a whisper.cpp model (no ggml magic).");
            }

            var values = new int[11];
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = ReadInt32(ref position);
            }

            HyperParameters = new WhisperCppHyperParameters(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7],
                values[8], values[9], values[10] % FileTypeQuantizationFactor);
            MelBins = ReadInt32(ref position);
            FrequencyBins = ReadInt32(ref position);
            MelFilters = Bytes(ref position, checked((long)MelBins * FrequencyBins * sizeof(float))).ToArray();
            int vocabulary = ReadInt32(ref position);
            var tokens = new byte[vocabulary][];
            for (int i = 0; i < vocabulary; i++)
            {
                tokens[i] = Bytes(ref position, ReadUInt32(ref position)).ToArray();
            }

            Tokens = tokens;
            Tensors = ReadTensors(ref position);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The hyper-parameters.</summary>
    public WhisperCppHyperParameters HyperParameters { get; }

    /// <summary>Mel bins of the stored filter bank.</summary>
    public int MelBins { get; }

    /// <summary>Frequency bins of the stored filter bank (FFT size / 2 + 1).</summary>
    public int FrequencyBins { get; }

    /// <summary>The filter bank, little-endian f32 [<see cref="MelBins"/>, <see cref="FrequencyBins"/>].</summary>
    public byte[] MelFilters { get; }

    /// <summary>The stored vocabulary as byte strings (the text tokens; special tokens are implied by the hyper-parameters).</summary>
    public IReadOnlyList<byte[]> Tokens { get; }

    /// <summary>The tensors by whisper.cpp name.</summary>
    public IReadOnlyDictionary<string, WhisperCppTensor> Tensors { get; }

    /// <summary>Opens and maps a file.</summary>
    /// <param name="path">The file.</param>
    /// <exception cref="InvalidDataException">The file is not a well-formed whisper.cpp model.</exception>
    public static WhisperCppFile Open(string path) => new(path);

    /// <summary>True when <paramref name="path"/> is a file starting with the ggml magic.</summary>
    /// <param name="path">A path.</param>
    public static bool HasMagic(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            using var stream = File.OpenRead(path);
            Span<byte> magic = stackalloc byte[4];
            return stream.Read(magic) == 4 && BinaryPrimitives.ReadUInt32LittleEndian(magic) == Magic;
        }
        catch (IOException)
        {
            return false;
        }
    }

    /// <summary>The address of a file offset; valid until disposal.</summary>
    /// <param name="offset">The file offset.</param>
    /// <exception cref="ObjectDisposedException">The file is disposed.</exception>
    public byte* Pointer(long offset)
    {
        ObjectDisposedException.ThrowIf(_base == null, this);
        return _base + offset;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_base != null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }

        _view?.Dispose();
        _file.Dispose();
    }

    private Dictionary<string, WhisperCppTensor> ReadTensors(ref long position)
    {
        var tensors = new Dictionary<string, WhisperCppTensor>(StringComparer.Ordinal);
        while (position < _length)
        {
            int dimensions = ReadInt32(ref position);
            int nameLength = ReadInt32(ref position);
            int typeId = ReadInt32(ref position);
            if (dimensions is < 1 or > 4 || nameLength is < 1 or > 1024)
            {
                throw new InvalidDataException($"Implausible tensor header at offset {position} (dimensions {dimensions}, name length {nameLength}).");
            }

            var dims = new long[dimensions];
            for (int i = 0; i < dimensions; i++)
            {
                // ggml lists the innermost dimension first; Sharplight shapes are row-major.
                dims[dimensions - 1 - i] = ReadInt32(ref position);
            }

            string name = System.Text.Encoding.UTF8.GetString(Bytes(ref position, nameLength));
            var type = TypeOf(typeId, name);
            var shape = new Shape(dims.Select(d => (Dim)d));
            long length = type.ByteSize(shape.Elements());
            tensors[name] = new WhisperCppTensor(name, type, shape, position, length);
            Bytes(ref position, length);
        }

        return tensors;
    }

    private static DataType TypeOf(int ggmlType, string name) => ggmlType switch
    {
        0 => DataType.F32,
        1 => DataType.F16,
        2 => DataType.Q4_0,
        3 => DataType.Q4_1,
        6 => DataType.Q5_0,
        7 => DataType.Q5_1,
        8 => DataType.Q8_0,
        10 => DataType.Q2_K,
        11 => DataType.Q3_K,
        12 => DataType.Q4_K,
        13 => DataType.Q5_K,
        14 => DataType.Q6_K,
        _ => throw new NotSupportedException($"whisper.cpp tensor '{name}': ggml type {ggmlType} is not supported."),
    };

    private ReadOnlySpan<byte> Bytes(ref long position, long count)
    {
        if (count < 0 || position + count > _length)
        {
            throw new InvalidDataException($"Truncated whisper.cpp file: {count} bytes at offset {position} exceed the file length {_length}.");
        }

        var span = new ReadOnlySpan<byte>(_base + position, checked((int)Math.Min(count, int.MaxValue)));
        position += count;
        return span;
    }

    private int ReadInt32(ref long position) => BinaryPrimitives.ReadInt32LittleEndian(Bytes(ref position, sizeof(int)));

    private uint ReadUInt32(ref long position) => BinaryPrimitives.ReadUInt32LittleEndian(Bytes(ref position, sizeof(uint)));
}
