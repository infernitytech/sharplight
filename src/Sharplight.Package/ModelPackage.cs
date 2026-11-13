using System.Buffers;
using System.Buffers.Binary;
using System.Text;
using Sharplight.Models;
using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>
/// Sharplight's native single-file model format (<c>.slm</c>): a built model (Sharplight IR graphs, states, properties, tokenizer
/// metadata, weights) from any source format, plus any number of optional precompiled artifact sets (device code and driver
/// binaries for any providers). Built to be memory mapped: weights are stored contiguously at 64 KiB-aligned offsets and read in
/// place, artifacts are served straight from the mapping.
/// </summary>
/// <remarks>
/// <para>Layout (little endian):</para>
/// <code>
/// header   (64 B)  magic "SHARPLM\0", u32 version, u32 header size, u64 section-table offset, u32 section count
/// sections         DEFN (Sharplight IR), WIDX (weight index), WDAT (weight bytes), ARTS (one per artifact set; KSET in version 1)
/// table            per section: u32 kind, u32 name length, u64 offset, u64 length, UTF-8 name
/// </code>
/// <para>An artifact set is a descriptor (string → string: provider, tier, kind and provider-defined keys, see
/// <see cref="ArtifactDescriptorKeys"/>) plus entries (key → bytes). Loading consults the sets that match a device's
/// <see cref="IComputeDevice.ArtifactTargets"/> and skips the others, so a package may carry code for several device profiles and
/// binaries for several device/driver pairs, or none at all.</para>
/// <para>The section table is written last, so sets compiled later (another GPU, a newer driver) are appended with
/// <see cref="AddArtifacts"/> without rewriting the weights; superseded sets become dead space that <see cref="Compact"/> removes.
/// Version 1 files (kernel sets in KSET sections) stay readable; their sets are read with tier <see cref="ArtifactTiers.Mixed"/>.
/// Version 3 stores the IR in DEFN layout 2 (section name <c>ir.2</c>: control-flow subgraphs and data-dependent symbols); a DEFN
/// section with an empty name has the layout of versions 1 and 2.</para>
/// <para>Weights are stored in their canonical source types (layout transforms a device needs are cheap and applied during
/// upload), so one file serves every device; only artifact sets are device specific. <see cref="ModelPackageReader"/> inspects a
/// package without loading it.</para>
/// </remarks>
public static class ModelPackage
{
    /// <summary>File extension of model packages, including the dot.</summary>
    public const string Extension = ".slm";
    /// <summary>Format version written by <see cref="Save"/> and every other write; versions 1 to this one can be opened.</summary>
    public const int Version = 3;
    internal static ReadOnlySpan<byte> Magic => "SHARPLM\0"u8;
    internal const int HeaderSize = 64;
    internal const long WeightAlignment = 64 * 1024;

    /// <summary>
    /// Writes a built model (<paramref name="definition"/>: all graphs, states, properties, tokenizer metadata and
    /// every weight the graphs use) and the precompiled artifact sets <paramref name="artifacts"/> (e.g. from
    /// <c>LlmModel.ExportCompilation()</c>) to a new package file. Works for any source format (GGUF, Hugging Face,
    /// ONNX, ...): loading a package needs neither the original importer nor an architecture adapter.
    /// </summary>
    /// <param name="path">Destination file; written through a temporary file, so an existing file is replaced atomically.</param>
    /// <param name="definition">The built model; its weights are read and copied into the package.</param>
    /// <param name="artifacts">Artifact sets to store; may be empty (sets without entries are skipped).</param>
    /// <param name="log">Receives progress messages, or <see langword="null"/>.</param>
    /// <exception cref="IOException">The destination is the package the definition's weights are read from.</exception>
    /// <exception cref="InvalidDataException">Two weights share a name, or a weight has a symbolic shape or a source of the wrong size.</exception>
    public static void Save(string path, ModelDefinition definition, IEnumerable<CompilationCache> artifacts, Action<string>? log = null)
    {
        if (definition.OwnedResources.OfType<PackageFile>().Any(f => Path.GetFullPath(f.Path) == Path.GetFullPath(path)))
        {
            throw new IOException($"Cannot overwrite the package being read ({path}); use AddArtifacts to store new artifact sets.");
        }

        WriteReplacing(path, fs =>
        {
            fs.Write(new byte[HeaderSize]);
            var sections = new List<PackageSection>();

            sections.Add(PackageWriter.WriteSection(fs, PackageSectionKind.Definition, DefinitionCodec.SectionName, w => DefinitionCodec.Write(w, definition)));

            // Every weight of the definition plus any the graphs reference beyond it (e.g. auxiliary files).
            var byName = new Dictionary<string, WeightInfo>(definition.Weights, StringComparer.Ordinal);
            foreach (var g in definition.Graphs.Values.SelectMany(DefinitionCodec.WithSubgraphs))
            {
                foreach (var v in g.Weights)
                {
                    if (byName.TryGetValue(v.Weight!.Name, out var known) ? !ReferenceEquals(known, v.Weight) && !SameWeight(known, v.Weight) : false)
                    {
                        throw new InvalidDataException($"Two different weights are named '{v.Weight.Name}'.");
                    }
                    else
                    {
                        byName.TryAdd(v.Weight.Name, v.Weight);
                    }
                }
            }
            // Weight index needs the data offsets: lay the data out first, after the (sized) index.
            var weights = byName.Values.OrderBy(w => w.Name, StringComparer.Ordinal).ToList();
            var index = new MemoryStream();
            long dataStart = 0;
            var offsets = new long[weights.Count];
            {
                // Index size does not depend on offset values, so write it once with zeros to measure.
                WeightIndexCodec.Write(new BinaryWriter(index, Encoding.UTF8, leaveOpen: true), weights, offsets);
                long indexOffset = PackageWriter.Align(fs.Position, PackageWriter.SectionAlignment);
                dataStart = PackageWriter.Align(indexOffset + index.Length, WeightAlignment);
                long pos = dataStart;
                for (int i = 0; i < weights.Count; i++)
                {
                    offsets[i] = pos;
                    pos = PackageWriter.Align(pos + weights[i].ByteSize, WeightAlignment);
                }
                index.SetLength(0);
                WeightIndexCodec.Write(new BinaryWriter(index, Encoding.UTF8, leaveOpen: true), weights, offsets);
                PackageWriter.Pad(fs, indexOffset);
                index.Position = 0;
                index.CopyTo(fs);
                sections.Add(new PackageSection(PackageSectionKind.WeightIndex, "", indexOffset, index.Length));
            }

            long total = weights.Sum(w => w.ByteSize);
            long written = 0;
            long nextLog = 0;
            var buf = ArrayPool<byte>.Shared.Rent(16 << 20);
            try
            {
                for (int i = 0; i < weights.Count; i++)
                {
                    PackageWriter.Pad(fs, offsets[i]);
                    CopyWeight(weights[i], fs, buf);
                    written += weights[i].ByteSize;
                    if (log is not null && written >= nextLog)
                    {
                        log($"  weights {written / 1e9:F1}/{total / 1e9:F1} GB");
                        nextLog += 4L << 30;
                    }
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buf);
            }
            // Without weights nothing was padded to dataStart: the (empty) data section starts where the index ends.
            long dataEnd = fs.Position;
            long dataOffset = weights.Count == 0 ? dataEnd : dataStart;
            sections.Add(new PackageSection(PackageSectionKind.WeightData, "", dataOffset, dataEnd - dataOffset));

            sections.AddRange(WriteArtifactSets(fs, artifacts));
            PackageWriter.WriteTable(fs, sections);
        });
    }

    /// <summary>
    /// Appends artifact sets to an existing package without rewriting anything else; a stored set with the same descriptor is
    /// replaced (its bytes become dead space, see <see cref="Compact"/>). Safe while the package is open for reading in this or
    /// another process: existing bytes are never modified except the header's table pointer, which is switched last. The package
    /// is written as <see cref="Version"/> afterwards.
    /// </summary>
    /// <param name="path">The existing package.</param>
    /// <param name="artifacts">Sets to store; sets without entries are skipped.</param>
    /// <returns>The ids (section names) of the stored sets.</returns>
    /// <exception cref="InvalidDataException">The file is not a supported package.</exception>
    public static IReadOnlyList<string> AddArtifacts(string path, IEnumerable<CompilationCache> artifacts)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
        var (_, sections) = ReadHeader(fs);
        var list = sections.ToList();
        fs.Seek(0, SeekOrigin.End);
        var added = WriteArtifactSets(fs, artifacts);
        list.RemoveAll(s => s.Kind == PackageSectionKind.ArtifactSet && added.Any(a => a.Name == s.Name));
        list.AddRange(added);
        PackageWriter.WriteTable(fs, list);
        return [.. added.Select(a => a.Name)];
    }

    /// <summary>
    /// Reads every artifact set of a package (including version 1 kernel sets, with their synthesized <see cref="ArtifactTiers.Mixed"/>
    /// descriptor) into memory, oldest first, e.g. to store them again with a converted model (<see cref="Save"/>).
    /// </summary>
    /// <param name="path">The existing package.</param>
    /// <exception cref="InvalidDataException">The file is not a supported package.</exception>
    public static IReadOnlyList<CompilationCache> ReadArtifactSets(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        var (_, sections) = ReadHeader(fs);
        var reader = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);
        var sets = new List<CompilationCache>();
        foreach (var section in sections.Where(s => s.IsArtifactSet))
        {
            fs.Seek(section.Offset, SeekOrigin.Begin);
            var directory = ArtifactSetCodec.ReadDirectory(reader, section);
            var set = new CompilationCache(directory.Descriptor);
            foreach (var (key, offset, length) in directory.Entries)
            {
                var bytes = new byte[length];
                fs.Seek(offset, SeekOrigin.Begin);
                fs.ReadExactly(bytes);
                set.Put(key, bytes);
            }

            sets.Add(set);
        }

        return sets;
    }

    /// <summary>
    /// Removes artifact sets (including version 1 kernel sets) from a package by rewriting only its section table; the removed
    /// bytes become dead space (see <see cref="Compact"/>). Safe while the package is open for reading.
    /// </summary>
    /// <param name="path">The existing package.</param>
    /// <param name="provider">Only sets of this provider, or <see langword="null"/> for every provider.</param>
    /// <param name="tier">Only sets of this tier (<see cref="ArtifactTiers"/>), or <see langword="null"/> for every tier.</param>
    /// <returns>The number of sets removed.</returns>
    /// <exception cref="InvalidDataException">The file is not a supported package.</exception>
    public static int Strip(string path, string? provider = null, string? tier = null)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.ReadWrite | FileShare.Delete);
        var (_, sections) = ReadHeader(fs);
        var reader = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);
        var removed = new HashSet<PackageSection>();
        foreach (var section in sections.Where(s => s.IsArtifactSet))
        {
            fs.Seek(section.Offset, SeekOrigin.Begin);
            var descriptor = ArtifactSetCodec.ReadDirectory(reader, section).Descriptor;
            if ((provider is null || ArtifactDescriptor.Get(descriptor, ArtifactDescriptorKeys.Provider) == provider)
                && (tier is null || ArtifactDescriptor.Get(descriptor, ArtifactDescriptorKeys.Tier) == tier))
            {
                removed.Add(section);
            }
        }

        if (removed.Count > 0)
        {
            PackageWriter.WriteTable(fs, [.. sections.Where(s => !removed.Contains(s))]);
        }

        return removed.Count;
    }

    /// <summary>
    /// Rewrites a package without dead space (superseded or stripped artifact sets, old section tables). The model part (definition,
    /// weight index and weights) is copied byte for byte, so weight offsets do not change; live artifact sets are packed after it.
    /// The file is replaced atomically; readers that have it open keep reading the old file.
    /// </summary>
    /// <param name="path">The existing package.</param>
    /// <returns>The number of bytes the file shrank by.</returns>
    /// <exception cref="InvalidDataException">The file is not a supported package.</exception>
    public static long Compact(string path)
    {
        long before = new FileInfo(path).Length;
        using (var source = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
        {
            var (_, sections) = ReadHeader(source);
            long modelEnd = sections.Where(s => !s.IsArtifactSet).Select(s => s.Offset + s.Length).DefaultIfEmpty(HeaderSize).Max();
            WriteReplacing(path, fs =>
            {
                Copy(source, 0, modelEnd, fs);
                var compacted = new List<PackageSection>(sections.Count);
                foreach (var s in sections)
                {
                    if (s.Offset + s.Length <= modelEnd)
                    {
                        compacted.Add(s);
                        continue;
                    }

                    // Artifact sections only hold section-relative offsets, so they move as a block.
                    compacted.Add(PackageWriter.WriteSection(fs, s.Kind, s.Name, w =>
                    {
                        w.Flush();
                        Copy(source, s.Offset, s.Length, fs);
                    }));
                }

                PackageWriter.WriteTable(fs, compacted);
            });
        }

        return before - new FileInfo(path).Length;
    }

    /// <summary>Writes <paramref name="path"/> through a temporary file that replaces it only when <paramref name="write"/> succeeded.</summary>
    private static void WriteReplacing(string path, Action<FileStream> write)
    {
        var tmp = path + ".tmp";
        try
        {
            using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.ReadWrite, FileShare.None, 1 << 20))
            {
                write(fs);
            }
            File.Move(tmp, path, overwrite: true);
        }
        catch
        {
            try
            {
                File.Delete(tmp);
            }
            catch (Exception)
            {
                // Best effort: the original failure is the one worth reporting.
            }

            throw;
        }
    }

    private static void Copy(FileStream source, long offset, long length, Stream destination)
    {
        var buffer = ArrayPool<byte>.Shared.Rent(16 << 20);
        try
        {
            source.Seek(offset, SeekOrigin.Begin);
            for (long done = 0; done < length;)
            {
                int n = source.Read(buffer, 0, (int)Math.Min(buffer.Length, length - done));
                if (n == 0)
                {
                    throw new InvalidDataException("Package section lies outside the file.");
                }

                destination.Write(buffer, 0, n);
                done += n;
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }

    private static List<PackageSection> WriteArtifactSets(FileStream fs, IEnumerable<CompilationCache> artifacts)
    {
        var written = new List<PackageSection>();
        foreach (var set in artifacts.Where(a => a.Entries.Count > 0))
        {
            var section = ArtifactSetCodec.Write(fs, set);
            written.RemoveAll(s => s.Name == section.Name);
            written.Add(section);
        }

        return written;
    }

    // ---- writing ---------------------------------------------------------------------------------------

    private static bool SameWeight(WeightInfo a, WeightInfo b) => ReferenceEquals(a.Source, b.Source) && a.Type == b.Type && a.Shape.Equals(b.Shape);

    private static unsafe void CopyWeight(WeightInfo w, FileStream fs, byte[] buf)
    {
        long size = w.ByteSize;
        if (w.Source.ByteLength < size)
        {
            throw new InvalidDataException($"Weight {w.Name}: source has {w.Source.ByteLength} bytes, expected {size}.");
        }

        nint ptr = w.Source.TryGetPointer();
        for (long done = 0; done < size;)
        {
            int n = (int)Math.Min(buf.Length, size - done);
            if (ptr != 0)
            {
                fs.Write(new ReadOnlySpan<byte>((byte*)ptr + done, n));
            }
            else
            {
                w.Source.Read(done, buf.AsSpan(0, n));
                fs.Write(buf, 0, n);
            }
            done += n;
        }
    }

    // ---- reading ---------------------------------------------------------------------------------------

    internal static (int Version, IReadOnlyList<PackageSection> Sections) ReadHeader(Stream s)
    {
        Span<byte> h = stackalloc byte[HeaderSize];
        s.Seek(0, SeekOrigin.Begin);
        s.ReadExactly(h);
        if (!h[..8].SequenceEqual(Magic))
        {
            throw new InvalidDataException("Not a Sharplight package (bad magic).");
        }

        int version = BinaryPrimitives.ReadInt32LittleEndian(h[8..]);
        if (version < 1 || version > Version)
        {
            throw new InvalidDataException($"Unsupported Sharplight package version {version} (supported: 1 to {Version}).");
        }

        long table = BinaryPrimitives.ReadInt64LittleEndian(h[16..]);
        int count = BinaryPrimitives.ReadInt32LittleEndian(h[24..]);
        s.Seek(table, SeekOrigin.Begin);
        var r = new BinaryReader(s, Encoding.UTF8, leaveOpen: true);
        var sections = new List<PackageSection>(count);
        for (int i = 0; i < count; i++)
        {
            var kind = (PackageSectionKind)r.ReadUInt32();
            int nameLen = r.ReadInt32();
            long off = r.ReadInt64();
            long len = r.ReadInt64();
            sections.Add(new PackageSection(kind, Encoding.UTF8.GetString(r.ReadBytes(nameLen)), off, len));
        }
        return (version, sections);
    }

    /// <summary>True when <paramref name="path"/> is a file starting with the package magic.</summary>
    public static bool IsPackage(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            Span<byte> m = stackalloc byte[8];
            return fs.Read(m) == 8 && m.SequenceEqual(Magic);
        }
        catch (IOException)
        {
            return false;
        }
        catch (UnauthorizedAccessException)
        {
            return false;
        }
    }
}
