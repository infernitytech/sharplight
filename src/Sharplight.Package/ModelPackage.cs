using System.Buffers;
using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Text;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Tensors;

namespace Sharplight.Package;

/// <summary>
/// Sharplight's native single-file model format (<c>.slm</c>): a built model (Sharplight IR graphs, states,
/// properties, tokenizer metadata, weights) from any source format, plus compiled artifacts for any number of device targets. Built to be memory mapped: weights are
/// stored contiguously at 64 KiB-aligned offsets and read in place, compiled kernels and driver pipeline caches are
/// served straight from the mapping.
/// </summary>
/// <remarks>
/// <para>Layout (little endian):</para>
/// <code>
/// header   (64 B)  magic "SHARPLM\0", u32 version, u32 header size, u64 section-table offset, u32 section count
/// sections         DEFN (Sharplight IR: graphs, states, properties, metadata), WIDX (weight index), WDAT (weight bytes), KSET (one per device target)
/// table            per section: u32 kind, u32 name length, u64 offset, u64 length, UTF-8 name
/// </code>
/// <para>The section table is written last, so targets compiled later (another GPU, a newer driver) are appended
/// with <see cref="AddTargets"/> without rewriting the weights. Superseded sections become dead space.</para>
/// <para>Weights are stored in their canonical source types (layout transforms a device needs are cheap and
/// applied during upload), so one file serves every device; only KSET sections are device specific.</para>
/// </remarks>
public static class ModelPackage
{
    public const string Extension = ".slm";
    public const int Version = 1;
    internal static ReadOnlySpan<byte> Magic => "SHARPLM\0"u8;
    internal const int HeaderSize = 64;
    internal const long WeightAlignment = 64 * 1024;

    internal enum SectionKind : uint { Definition = 1, WeightIndex = 2, WeightData = 3, KernelSet = 4 }

    internal readonly record struct Section(SectionKind Kind, string Name, long Offset, long Length);

    /// <summary>
    /// Writes a built model (<paramref name="definition"/>: all graphs, states, properties, tokenizer metadata and
    /// every weight the graphs use) and the compilation caches of <paramref name="targets"/> (e.g. from
    /// <c>LlmModel.ExportCompilation()</c>) to a new package file. Works for any source format (GGUF, Hugging Face,
    /// ONNX, ...): loading a package needs neither the original importer nor an architecture adapter.
    /// </summary>
    public static void Save(string path, ModelDefinition definition, IEnumerable<CompilationCache> targets, Action<string>? log = null)
    {
        if (definition.OwnedResources.OfType<PackageFile>().Any(f => Path.GetFullPath(f.Path) == Path.GetFullPath(path)))
            throw new IOException($"Cannot overwrite the package being read ({path}); use AddTargets to store new targets.");
        var tmp = path + ".tmp";
        try
        {
            using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.ReadWrite, FileShare.None, 1 << 20))
            {
                fs.Write(new byte[HeaderSize]);
                var sections = new List<Section>();

                sections.Add(WriteSection(fs, SectionKind.Definition, "", w => DefinitionCodec.Write(w, definition)));

                // Every weight of the definition plus any the graphs reference beyond it (e.g. auxiliary files).
                var byName = new Dictionary<string, WeightInfo>(definition.Weights, StringComparer.Ordinal);
                foreach (var g in definition.Graphs.Values)
                    foreach (var v in g.Weights)
                        if (byName.TryGetValue(v.Weight!.Name, out var known) ? !ReferenceEquals(known, v.Weight) && !SameWeight(known, v.Weight) : false)
                            throw new InvalidDataException($"Two different weights are named '{v.Weight.Name}'.");
                        else byName.TryAdd(v.Weight.Name, v.Weight);
                // Weight index needs the data offsets: lay the data out first, after the (sized) index.
                var weights = byName.Values.OrderBy(w => w.Name, StringComparer.Ordinal).ToList();
                var index = new MemoryStream();
                long dataStart = 0;
                var offsets = new long[weights.Count];
                {
                    // Index size does not depend on offset values, so write it once with zeros to measure.
                    WriteIndex(new BinaryWriter(index, Encoding.UTF8, leaveOpen: true), weights, offsets);
                    long indexOffset = Align(fs.Position, 16);
                    dataStart = Align(indexOffset + index.Length, WeightAlignment);
                    long pos = dataStart;
                    for (int i = 0; i < weights.Count; i++)
                    {
                        offsets[i] = pos;
                        pos = Align(pos + weights[i].ByteSize, WeightAlignment);
                    }
                    index.SetLength(0);
                    WriteIndex(new BinaryWriter(index, Encoding.UTF8, leaveOpen: true), weights, offsets);
                    Pad(fs, indexOffset);
                    index.Position = 0;
                    index.CopyTo(fs);
                    sections.Add(new Section(SectionKind.WeightIndex, "", indexOffset, index.Length));
                }

                long total = weights.Sum(w => w.ByteSize), written = 0, nextLog = 0;
                var buf = ArrayPool<byte>.Shared.Rent(16 << 20);
                try
                {
                    for (int i = 0; i < weights.Count; i++)
                    {
                        Pad(fs, offsets[i]);
                        CopyWeight(weights[i], fs, buf);
                        written += weights[i].ByteSize;
                        if (log is not null && written >= nextLog) { log($"  weights {written / 1e9:F1}/{total / 1e9:F1} GB"); nextLog += 4L << 30; }
                    }
                }
                finally { ArrayPool<byte>.Shared.Return(buf); }
                long dataEnd = fs.Position;
                sections.Add(new Section(SectionKind.WeightData, "", dataStart, dataEnd - dataStart));

                foreach (var t in targets)
                    if (t.Entries.Count > 0)
                        sections.Add(WriteKernelSet(fs, t));
                WriteTable(fs, sections);
            }
            File.Move(tmp, path, overwrite: true);
        }
        catch
        {
            try { File.Delete(tmp); } catch { }
            throw;
        }
    }

    /// <summary>
    /// Appends compilation caches to an existing package (replacing stored ones of the same target). Safe while
    /// the package is open for reading in this or another process: existing bytes are never modified except the
    /// header's table pointer, which is switched last.
    /// </summary>
    public static void AddTargets(string path, IEnumerable<CompilationCache> targets)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
        var (_, sections) = ReadHeader(fs);
        var list = sections.ToList();
        fs.Seek(0, SeekOrigin.End);
        foreach (var t in targets)
        {
            if (t.Entries.Count == 0) continue;
            list.RemoveAll(s => s.Kind == SectionKind.KernelSet && s.Name == t.Target);
            list.Add(WriteKernelSet(fs, t));
        }
        WriteTable(fs, list);
    }

    // ---- writing ---------------------------------------------------------------------------------------

    static bool SameWeight(WeightInfo a, WeightInfo b) => ReferenceEquals(a.Source, b.Source) && a.Type == b.Type && a.Shape.Equals(b.Shape);

    static void WriteIndex(BinaryWriter w, List<WeightInfo> weights, long[] offsets)
    {
        w.Write(weights.Count);
        for (int i = 0; i < weights.Count; i++)
        {
            var wi = weights[i];
            MetadataCodec.WriteString(w, wi.Name);
            MetadataCodec.WriteString(w, wi.Type.Name);
            var dims = wi.Shape.Dims;
            w.Write(dims.Length);
            foreach (var d in dims) w.Write(d.IsSymbolic ? throw new InvalidDataException($"Weight {wi.Name} has a symbolic shape.") : d.Fixed);
            w.Write(offsets[i]);
            w.Write(wi.ByteSize);
        }
        w.Flush();
    }

    static unsafe void CopyWeight(WeightInfo w, FileStream fs, byte[] buf)
    {
        long size = w.ByteSize;
        if (w.Source.ByteLength < size) throw new InvalidDataException($"Weight {w.Name}: source has {w.Source.ByteLength} bytes, expected {size}.");
        nint ptr = w.Source.TryGetPointer();
        for (long done = 0; done < size;)
        {
            int n = (int)Math.Min(buf.Length, size - done);
            if (ptr != 0) fs.Write(new ReadOnlySpan<byte>((byte*)ptr + done, n));
            else
            {
                w.Source.Read(done, buf.AsSpan(0, n));
                fs.Write(buf, 0, n);
            }
            done += n;
        }
    }

    static Section WriteKernelSet(FileStream fs, CompilationCache cache)
    {
        var entries = cache.Entries.OrderBy(e => e.Key, StringComparer.Ordinal).ToList();
        return WriteSection(fs, SectionKind.KernelSet, cache.Target, w =>
        {
            // Directory first (offsets relative to the section start), then 16-byte aligned blobs.
            var dir = new MemoryStream();
            var dw = new BinaryWriter(dir, Encoding.UTF8, leaveOpen: true);
            dw.Write(entries.Count);
            foreach (var e in entries) { MetadataCodec.WriteString(dw, e.Key); dw.Write(0L); dw.Write(0L); }
            dw.Flush();
            long pos = Align(dir.Length, 16);
            var offs = new long[entries.Count];
            for (int i = 0; i < entries.Count; i++) { offs[i] = pos; pos = Align(pos + entries[i].Value.Length, 16); }
            dir.SetLength(0);
            dw.Write(entries.Count);
            for (int i = 0; i < entries.Count; i++) { MetadataCodec.WriteString(dw, entries[i].Key); dw.Write(offs[i]); dw.Write((long)entries[i].Value.Length); }
            dw.Flush();
            w.Write(dir.GetBuffer(), 0, (int)dir.Length);
            long written = dir.Length;
            for (int i = 0; i < entries.Count; i++)
            {
                w.Write(new byte[offs[i] - written]);
                w.Write(entries[i].Value.Span);
                written = offs[i] + entries[i].Value.Length;
            }
        });
    }

    static Section WriteSection(FileStream fs, SectionKind kind, string name, Action<BinaryWriter> body)
    {
        long start = Align(fs.Position, 16);
        Pad(fs, start);
        using (var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true)) { body(w); w.Flush(); }
        return new Section(kind, name, start, fs.Position - start);
    }

    static void WriteTable(FileStream fs, List<Section> sections)
    {
        long table = Align(fs.Seek(0, SeekOrigin.End), 16);
        Pad(fs, table);
        using (var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true))
        {
            foreach (var s in sections)
            {
                var name = Encoding.UTF8.GetBytes(s.Name);
                w.Write((uint)s.Kind);
                w.Write(name.Length);
                w.Write(s.Offset);
                w.Write(s.Length);
                w.Write(name);
            }
        }
        fs.Flush(flushToDisk: true);
        // Switch the header to the new table only after the table is durable.
        Span<byte> h = stackalloc byte[HeaderSize];
        Magic.CopyTo(h);
        BinaryPrimitives.WriteInt32LittleEndian(h[8..], Version);
        BinaryPrimitives.WriteInt32LittleEndian(h[12..], HeaderSize);
        BinaryPrimitives.WriteInt64LittleEndian(h[16..], table);
        BinaryPrimitives.WriteInt32LittleEndian(h[24..], sections.Count);
        fs.Seek(0, SeekOrigin.Begin);
        fs.Write(h);
        fs.Flush(flushToDisk: true);
    }

    static void Pad(FileStream fs, long to)
    {
        if (fs.Position > to) throw new InvalidOperationException("Package writer position overran its layout.");
        if (fs.Position < to) fs.Write(new byte[to - fs.Position]);
    }

    internal static long Align(long v, long a) => (v + a - 1) / a * a;

    // ---- reading ---------------------------------------------------------------------------------------

    internal static (int Version, IReadOnlyList<Section> Sections) ReadHeader(Stream s)
    {
        Span<byte> h = stackalloc byte[HeaderSize];
        s.Seek(0, SeekOrigin.Begin);
        s.ReadExactly(h);
        if (!h[..8].SequenceEqual(Magic)) throw new InvalidDataException("Not a Sharplight package (bad magic).");
        int version = BinaryPrimitives.ReadInt32LittleEndian(h[8..]);
        if (version != Version) throw new InvalidDataException($"Unsupported Sharplight package version {version} (expected {Version}).");
        long table = BinaryPrimitives.ReadInt64LittleEndian(h[16..]);
        int count = BinaryPrimitives.ReadInt32LittleEndian(h[24..]);
        s.Seek(table, SeekOrigin.Begin);
        var r = new BinaryReader(s, Encoding.UTF8, leaveOpen: true);
        var sections = new List<Section>(count);
        for (int i = 0; i < count; i++)
        {
            var kind = (SectionKind)r.ReadUInt32();
            int nameLen = r.ReadInt32();
            long off = r.ReadInt64(), len = r.ReadInt64();
            sections.Add(new Section(kind, Encoding.UTF8.GetString(r.ReadBytes(nameLen)), off, len));
        }
        return (version, sections);
    }

    /// <summary>True when <paramref name="path"/> is a file starting with the package magic.</summary>
    public static bool IsPackage(string path)
    {
        if (!File.Exists(path)) return false;
        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            Span<byte> m = stackalloc byte[8];
            return fs.Read(m) == 8 && m.SequenceEqual(Magic);
        }
        catch (IOException) { return false; }
        catch (UnauthorizedAccessException) { return false; }
    }
}

/// <summary>An open, memory-mapped package.</summary>
public sealed unsafe class PackageFile : IDisposable, ICompilationCacheStore
{
    readonly MemoryMappedFile _mmf;
    readonly MemoryMappedViewAccessor _view;
    readonly byte* _base;
    readonly long _length;
    readonly Dictionary<string, KernelSet> _targets = new(StringComparer.Ordinal);
    bool _disposed;

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
            if (s.Offset < 0 || s.Length < 0 || s.Offset + s.Length > _length)
                throw new InvalidDataException($"Package section {s.Kind} '{s.Name}' lies outside the file.");
        var idx = sections.Single(s => s.Kind == ModelPackage.SectionKind.WeightIndex);
        Weights = ReadIndex(Reader(idx));
        var def = sections.Single(s => s.Kind == ModelPackage.SectionKind.Definition);
        Definition = DefinitionCodec.Read(Reader(def), Weights, [this]);
        foreach (var s in sections.Where(s => s.Kind == ModelPackage.SectionKind.KernelSet))
            _targets[s.Name] = new KernelSet(this, s);
    }

    BinaryReader Reader(ModelPackage.Section s)
        => new(new UnmanagedMemoryStream(_base + s.Offset, s.Length), Encoding.UTF8);

    Dictionary<string, WeightInfo> ReadIndex(BinaryReader r)
    {
        int n = r.ReadInt32();
        var d = new Dictionary<string, WeightInfo>(n, StringComparer.Ordinal);
        for (int i = 0; i < n; i++)
        {
            string name = MetadataCodec.ReadString(r);
            var type = DataType.Get(MetadataCodec.ReadString(r));
            int rank = r.ReadInt32();
            var dims = new Dim[rank];
            for (int j = 0; j < rank; j++) dims[j] = Dim.Of(r.ReadInt64());
            long off = r.ReadInt64(), len = r.ReadInt64();
            if (off < 0 || off + len > _length) throw new InvalidDataException($"Weight {name} lies outside the package.");
            var shape = new Shape(dims);
            if (type.ByteSize(shape.Elements()) != len) throw new InvalidDataException($"Weight {name}: {len} bytes do not match {type}{shape}.");
            d[name] = new WeightInfo { Name = name, Type = type, Shape = shape, Source = new MappedSource(this, off, len) };
        }
        return d;
    }

    public ICompilationCache? Find(string target) => _targets.GetValueOrDefault(target);

    void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(_disposed, this);

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _view.SafeMemoryMappedViewHandle.ReleasePointer();
        _view.Dispose();
        _mmf.Dispose();
    }

    sealed class MappedSource(PackageFile file, long offset, long length) : IWeightSource
    {
        public long ByteLength => length;
        public void Read(long o, Span<byte> destination)
        {
            file.ThrowIfDisposed();
            if (o < 0 || o + destination.Length > length) throw new ArgumentOutOfRangeException(nameof(o));
            new ReadOnlySpan<byte>(file._base + offset + o, destination.Length).CopyTo(destination);
        }
        public nint TryGetPointer()
        {
            file.ThrowIfDisposed();
            return (nint)(file._base + offset);
        }
    }

    /// <summary>A stored target's entries, served from the mapping without copying.</summary>
    sealed class KernelSet : ICompilationCache
    {
        readonly PackageFile _file;
        readonly Dictionary<string, (long Offset, int Length)> _entries = new(StringComparer.Ordinal);

        public KernelSet(PackageFile file, ModelPackage.Section s)
        {
            _file = file;
            var r = file.Reader(s);
            int n = r.ReadInt32();
            for (int i = 0; i < n; i++)
            {
                string key = MetadataCodec.ReadString(r);
                long off = r.ReadInt64(), len = r.ReadInt64();
                if (off < 0 || len < 0 || len > int.MaxValue || off + len > s.Length) throw new InvalidDataException($"Package entry {key} lies outside its section.");
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

    sealed class MappedMemory(byte* ptr, int length) : MemoryManager<byte>
    {
        public override Span<byte> GetSpan() => new(ptr, length);
        public override MemoryHandle Pin(int elementIndex = 0) => new(ptr + elementIndex);
        public override void Unpin() { }
        protected override void Dispose(bool disposing) { }
    }
}

/// <summary>Opens <c>.slm</c> packages through <see cref="ModelImporterRegistry"/>.</summary>
public sealed class ModelPackageImporter : IModelImporter
{
    public string Format => "sharplight";

    public static void Register() => ModelImporterRegistry.Register(new ModelPackageImporter());

    public bool CanOpen(string path) => ModelPackage.IsPackage(path);

    public ModelSource Open(string path)
    {
        var file = new PackageFile(path);
        try
        {
            return new ModelSource
            {
                Format = Format,
                Architecture = file.Definition.Architecture,
                Metadata = file.Definition.SourceMetadata,
                Weights = file.Weights,
                Definition = file.Definition,
                OwnedResources = [file],
                Path = path,
                CompilationCaches = file,
            };
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }
}
