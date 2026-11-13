using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Text;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Package;

/// <summary>An open, memory-mapped package.</summary>
internal sealed unsafe class PackageFile : IDisposable, ICompilationCacheStore
{
    private readonly MemoryMappedFile _mmf;
    private readonly MemoryMappedViewAccessor _view;
    private readonly byte* _base;
    private readonly long _length;
    private readonly Dictionary<string, KernelSet> _targets = new(StringComparer.Ordinal);
    private bool _disposed;

    public string Path { get; }
    public IReadOnlyDictionary<string, WeightInfo> Weights { get; }
    /// <summary>The stored model; owns this file (disposing the definition's resources closes it).</summary>
    public ModelDefinition Definition { get; }
    public IEnumerable<string> Targets => _targets.Keys;

    public PackageFile(string path)
    {
        Path = path;
        var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        IReadOnlyList<ModelPackage.Section> sections;
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

        var idx = sections.Single(s => s.Kind == ModelPackage.SectionKind.WeightIndex);
        Weights = ReadIndex(Reader(idx));
        var def = sections.Single(s => s.Kind == ModelPackage.SectionKind.Definition);
        Definition = DefinitionCodec.Read(Reader(def), Weights, [this]);
        foreach (var s in sections.Where(s => s.Kind == ModelPackage.SectionKind.KernelSet))
        {
            _targets[s.Name] = new KernelSet(this, s);
        }
    }

    private BinaryReader Reader(ModelPackage.Section s)
        => new(new UnmanagedMemoryStream(_base + s.Offset, s.Length), Encoding.UTF8);

    private Dictionary<string, WeightInfo> ReadIndex(BinaryReader r)
    {
        int n = r.ReadInt32();
        var d = new Dictionary<string, WeightInfo>(n, StringComparer.Ordinal);
        for (int i = 0; i < n; i++)
        {
            string name = MetadataCodec.ReadString(r);
            var type = DataType.Get(MetadataCodec.ReadString(r));
            int rank = r.ReadInt32();
            var dims = new Dim[rank];
            for (int j = 0; j < rank; j++)
            {
                dims[j] = Dim.Of(r.ReadInt64());
            }

            long off = r.ReadInt64(), len = r.ReadInt64();
            if (off < 0 || off + len > _length)
            {
                throw new InvalidDataException($"Weight {name} lies outside the package.");
            }

            var shape = new Shape(dims);
            if (type.ByteSize(shape.Elements()) != len)
            {
                throw new InvalidDataException($"Weight {name}: {len} bytes do not match {type}{shape}.");
            }

            d[name] = new WeightInfo { Name = name, Type = type, Shape = shape, Source = new MappedSource(this, off, len) };
        }
        return d;
    }

    public ICompilationCache? Find(string target) => _targets.GetValueOrDefault(target);

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

    /// <summary>A stored target's entries, served from the mapping without copying.</summary>
    private sealed class KernelSet : ICompilationCache
    {
        private readonly PackageFile _file;
        private readonly Dictionary<string, (long Offset, int Length)> _entries = new(StringComparer.Ordinal);

        public KernelSet(PackageFile file, ModelPackage.Section s)
        {
            _file = file;
            var r = file.Reader(s);
            int n = r.ReadInt32();
            for (int i = 0; i < n; i++)
            {
                string key = MetadataCodec.ReadString(r);
                long off = r.ReadInt64(), len = r.ReadInt64();
                if (off < 0 || len < 0 || len > int.MaxValue || off + len > s.Length)
                {
                    throw new InvalidDataException($"Package entry {key} lies outside its section.");
                }

                _entries[key] = (s.Offset + off, (int)len);
            }
        }

        public bool TryGet(string key, out ReadOnlyMemory<byte> data)
        {
            _file.ThrowIfDisposed();
            if (!_entries.TryGetValue(key, out var e)) { data = default; return false; }
            data = new MappedMemory(_file._base + e.Offset, e.Length).Memory;
            return true;
        }

        public void Put(string key, ReadOnlyMemory<byte> data) => throw new NotSupportedException("Stored package targets are read-only.");
    }

    private sealed class MappedMemory(byte* ptr, int length) : MemoryManager<byte>
    {
        public override Span<byte> GetSpan() => new(ptr, length);
        public override MemoryHandle Pin(int elementIndex = 0) => new(ptr + elementIndex);
        public override void Unpin() { }
        protected override void Dispose(bool disposing) { }
    }
}
