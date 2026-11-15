using System.IO.MemoryMappedFiles;

namespace Sharplight.Import.Onnx;

/// <summary>A read-only memory-mapped file whose address stays valid until disposed.</summary>
internal sealed unsafe class MappedFile : IDisposable
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _view;
    private byte* _base;
    public string Path { get; }
    public long Length { get; }
    public byte* Pointer => _base;

    public MappedFile(string path)
    {
        Path = path;
        // The stream's length is the target's when the path is a symbolic link (FileInfo reports the link's own size).
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Length = stream.Length;
        if (Length == 0)
        {
            stream.Dispose();
            throw new InvalidDataException($"{path} is empty.");
        }

        _mmf = MemoryMappedFile.CreateFromFile(stream, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
        _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref _base);
        _base += _view.PointerOffset;
    }

    public ReadOnlySpan<byte> AsSpan(long offset = 0, long length = -1)
    {
        if (length < 0)
        {
            length = Length - offset;
        }

        return new ReadOnlySpan<byte>(_base + offset, checked((int)length));
    }

    public void Dispose()
    {
        if (_base != null)
        {
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
            _base = null;
        }
        _view.Dispose();
        _mmf.Dispose();
    }
}
