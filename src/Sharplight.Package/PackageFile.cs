using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Text;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>An open, memory-mapped package.</summary>
internal sealed unsafe class PackageFile : IDisposable, ICompilationCacheStore
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _base;
    private readonly long _length;
    private bool _disposed;

    public string Path { get; }
    public IReadOnlyDictionary<string, WeightInfo> Weights { get; }
    /// <summary>The stored model; owns this file (disposing the definition's resources closes it).</summary>
    public ModelDefinition Definition { get; }
    /// <summary>Stored artifact sets (ARTS and legacy KSET sections), newest first.</summary>
    public IReadOnlyList<StoredArtifactSet> ArtifactSets { get; }

    public PackageFile(string path)
    {
        Path = path;
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        IReadOnlyList<PackageSection> sections;
        try
        {
            (_, sections) = ModelPackage.ReadHeader(fs);
            _length = fs.Length;
            _mmf = MemoryMappedFile.CreateFromFile(fs, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
        }
        catch
        {
            fs.Dispose();
            throw;
        }
        _view = _mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* p = null;
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        _base = p + _view.PointerOffset;

        foreach (var s in sections)
        {
            if (s.Offset < 0 || s.Length < 0 || s.Offset + s.Length > _length)
            {
                throw new InvalidDataException($"Package section {s.Kind} '{s.Name}' lies outside the file.");
            }
        }

        var idx = sections.Single(s => s.Kind == PackageSectionKind.WeightIndex);
        Weights = ReadIndex(Reader(idx));
        var def = sections.Single(s => s.Kind == PackageSectionKind.Definition);
        Definition = DefinitionCodec.Read(Reader(def), Weights, [this]);
        var sets = new List<StoredArtifactSet>();
        foreach (var s in sections.Where(s => s.IsArtifactSet).Reverse())
        {
            var directory = ArtifactSetCodec.ReadDirectory(Reader(s), s);
            sets.Add(new StoredArtifactSet(directory.Descriptor, new MappedArtifactSet(this, directory)));
        }
        ArtifactSets = sets;
    }

    private BinaryReader Reader(PackageSection s)
        => new(new UnmanagedMemoryStream(_base + s.Offset, s.Length), Encoding.UTF8);

    private Dictionary<string, WeightInfo> ReadIndex(BinaryReader r)
    {
        var entries = WeightIndexCodec.Read(r, _length);
        var weights = new Dictionary<string, WeightInfo>(entries.Count, StringComparer.Ordinal);
        foreach (var e in entries)
        {
            weights[e.Name] = new WeightInfo { Name = e.Name, Type = e.Type, Shape = e.Shape, Source = new MappedSource(this, e.Offset, e.ByteSize) };
        }

        return weights;
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _mmf.Dispose();
    }

    private sealed class MappedSource(PackageFile file, long offset, long length) : IWeightSource
    {
        public long ByteLength => length;
        public void Read(long o, Span<byte> destination)
        {
            file.ThrowIfDisposed();
            if (o < 0 || o + destination.Length > length)
            {
                throw new ArgumentOutOfRangeException(nameof(o));
            }

            new ReadOnlySpan<byte>(file._base + offset + o, destination.Length).CopyTo(destination);
        }
        public nint TryGetPointer()
        {
            file.ThrowIfDisposed();
            return (nint)(file._base + offset);
        }
    }

    /// <summary>A stored set's entries, served from the mapping without copying.</summary>
    private sealed class MappedArtifactSet : ICompilationCache
    {
        private readonly PackageFile _file;
        private readonly Dictionary<string, (long Offset, int Length)> _entries = new(StringComparer.Ordinal);

        public MappedArtifactSet(PackageFile file, ArtifactSetDirectory directory)
        {
            _file = file;
            foreach (var (key, offset, length) in directory.Entries)
            {
                _entries[key] = (offset, length);
            }
        }

        public bool TryGet(string key, out ReadOnlyMemory<byte> data)
        {
            _file.ThrowIfDisposed();
            if (!_entries.TryGetValue(key, out var e))
            {
                data = default;
                return false;
            }
            data = new MappedMemory(_file._base + e.Offset, e.Length).Memory;
            return true;
        }

        public void Put(string key, ReadOnlyMemory<byte> data) => throw new NotSupportedException("Stored artifact sets are read-only.");
    }

    private sealed class MappedMemory(byte* ptr, int length) : MemoryManager<byte>
    {
        public override Span<byte> GetSpan() => new(ptr, length);
        public override MemoryHandle Pin(int elementIndex = 0) => new(ptr + elementIndex);
        public override void Unpin() { }
        protected override void Dispose(bool disposing) { }
    }
}
