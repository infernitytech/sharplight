using System.IO.MemoryMappedFiles;

namespace Sharplight.Import.PyTorch;

/// <summary>
/// A memory-mapped PyTorch checkpoint (<c>.bin</c>, <c>.pt</c>, <c>.pth</c>, <c>.ckpt</c>; zip or legacy format). Opening parses only
/// the pickle, with a restricted unpickler that never executes code (see <see cref="UnsafePickleException"/>); tensor bytes stay in the
/// map and are addressed by <see cref="TorchTensorEntry.FileOffset"/>. Disposing unmaps the file.
/// </summary>
public sealed unsafe class TorchCheckpointFile : IDisposable
{
    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _view;
    private byte* _base;

    private TorchCheckpointFile(string path, TorchCheckpointOptions options)
    {
        Path = path;
        var fileInfo = new FileInfo(path);
        Length = (fileInfo.ResolveLinkTarget(returnFinalTarget: true) as FileInfo ?? fileInfo).Length; // a symbolic link reports its own size
        if (Length == 0)
        {
            throw new InvalidDataException($"{path}: the file is empty.");
        }

        _mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        try
        {
            _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            _base = pointer + _view.PointerOffset;
            Tensors = TorchCheckpointReader.Read(new FileView(_base, Length), options);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>The path the file was opened from.</summary>
    public string Path { get; }

    /// <summary>Size of the file in bytes.</summary>
    public long Length { get; }

    /// <summary>The tensors by state-dict key, in file order.</summary>
    public IReadOnlyDictionary<string, TorchTensorEntry> Tensors { get; } = new Dictionary<string, TorchTensorEntry>();

    /// <summary>The start of the file mapping; throws <see cref="ObjectDisposedException"/> after disposal.</summary>
    public byte* BasePointer
    {
        get
        {
            ObjectDisposedException.ThrowIf(_base == null, this);
            return _base;
        }
    }

    /// <summary>Maps <paramref name="path"/> and indexes its tensors.</summary>
    /// <param name="path">The checkpoint file.</param>
    /// <param name="options">Reader options; null for <see cref="TorchCheckpointOptions.Default"/>.</param>
    /// <exception cref="UnsafePickleException">The pickle uses an opcode or global outside the tensor state-dict subset.</exception>
    /// <exception cref="InvalidDataException">The file is malformed.</exception>
    /// <exception cref="NotSupportedException">The file has a non-contiguous tensor, is big-endian, or uses an unsupported legacy layout.</exception>
    public static TorchCheckpointFile Open(string path, TorchCheckpointOptions? options = null) => new(path, options ?? TorchCheckpointOptions.Default);

    /// <summary>Unmaps the file. Safe to call more than once.</summary>
    public void Dispose()
    {
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
