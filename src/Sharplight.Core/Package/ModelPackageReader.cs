using System.Buffers.Binary;
using System.Text;
using Sharplight.Models;

namespace Sharplight.Package;

/// <summary>
/// Inspects package files without loading them: reads the header, section table, weight index, the definition section and the
/// artifact directories, but no weight bytes and no artifact entry bytes. For viewers and tools (<c>sharplight-convert info</c>).
/// </summary>
public static class ModelPackageReader
{
    /// <summary>Reads the structure of a package.</summary>
    /// <param name="path">A package file (version 1 to 3).</param>
    /// <returns>A snapshot of the package; the file is closed again.</returns>
    /// <exception cref="InvalidDataException">The file is not a supported package or is damaged.</exception>
    public static PackageInfo Open(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        var (version, sections) = ModelPackage.ReadHeader(fs);
        long length = fs.Length;
        foreach (var s in sections)
        {
            if (s.Offset < 0 || s.Length < 0 || s.Offset + s.Length > length)
            {
                throw new InvalidDataException($"Package section {s.Tag} '{s.Name}' lies outside the file.");
            }
        }

        var reader = new BinaryReader(fs, Encoding.UTF8, leaveOpen: true);
        var index = sections.Single(s => s.Kind == PackageSectionKind.WeightIndex);
        fs.Seek(index.Offset, SeekOrigin.Begin);
        var weights = WeightIndexCodec.Read(reader, length);

        var definitionSection = sections.Single(s => s.Kind == PackageSectionKind.Definition);
        fs.Seek(definitionSection.Offset, SeekOrigin.Begin);
        var definitionBytes = reader.ReadBytes(checked((int)definitionSection.Length));

        ModelDescription? description = null;
        foreach (var s in sections.Where(s => s.Kind == PackageSectionKind.Description))
        {
            fs.Seek(s.Offset, SeekOrigin.Begin);
            description = ModelDescription.FromJson(Encoding.UTF8.GetString(reader.ReadBytes(checked((int)s.Length))));
        }

        var attachments = sections.Where(s => s.Kind == PackageSectionKind.Attachment)
            .Select(s => new PackageAttachmentInfo(s.Name, s.Offset, s.Length)).ToList();

        var artifactSets = new List<ArtifactSetInfo>();
        foreach (var s in sections.Where(s => s.IsArtifactSet))
        {
            fs.Seek(s.Offset, SeekOrigin.Begin);
            artifactSets.Add(new ArtifactSetInfo(ArtifactSetCodec.ReadDirectory(reader, s)));
        }

        Span<byte> header = stackalloc byte[ModelPackage.HeaderSize];
        fs.Seek(0, SeekOrigin.Begin);
        fs.ReadExactly(header);
        long tableOffset = BinaryPrimitives.ReadInt64LittleEndian(header[16..]);
        long tableLength = sections.Sum(s => SectionTableEntryBytes + Encoding.UTF8.GetByteCount(s.Name));

        return new PackageInfo(
            path,
            length,
            version,
            [.. sections.Select(s => new PackageSectionInfo(s.Tag, s.Name, s.Offset, s.Length))],
            weights,
            attachments,
            artifactSets,
            CompactedSize(sections),
            tableOffset,
            tableLength,
            () => DecodeDefinition(definitionBytes, weights, definitionSection.Name, description, attachments));
    }

    /// <summary>
    /// The kind tags of every section kind the format defines, in kind-number order: <c>DEFN</c>, <c>WIDX</c>, <c>WDAT</c>,
    /// <c>KSET</c> (version 1), <c>ARTS</c>, <c>DESC</c> and <c>ATCH</c>. A <see cref="PackageSectionInfo.Kind"/> that is not in this list is a section of a
    /// newer format version (its number), which readers skip.
    /// </summary>
    public static IReadOnlyList<string> SectionKinds { get; } =
        [.. Enum.GetValues<PackageSectionKind>().Select(kind => new PackageSection(kind, "", 0, 0).Tag)];

    /// <summary>Bytes of one section table entry without its name: kind, name length, offset and length.</summary>
    private const int SectionTableEntryBytes = 24;

    private static DefinitionInfo DecodeDefinition(byte[] bytes, IReadOnlyList<PackageWeightInfo> weights, string sectionName,
        ModelDescription? description, IReadOnlyList<PackageAttachmentInfo> attachments)
    {
        var detached = weights.ToDictionary(
            w => w.Name,
            w => new WeightInfo { Name = w.Name, Type = w.Type, Shape = w.Shape, Source = new DetachedWeightSource(w.ByteSize) },
            StringComparer.Ordinal);
        using var reader = new BinaryReader(new MemoryStream(bytes, writable: false), Encoding.UTF8);
        var detachedAttachments = attachments.ToDictionary(a => a.Name, a => (IWeightSource)new DetachedWeightSource(a.Length), StringComparer.Ordinal);
        return new DefinitionInfo(DefinitionCodec.Read(reader, detached, [], sectionName, description, detachedAttachments));
    }

    /// <summary>Size of the file <see cref="ModelPackage.Compact"/> would write (same layout rules).</summary>
    private static long CompactedSize(IReadOnlyList<PackageSection> sections)
    {
        long end = sections.Where(s => !s.IsArtifactSet).Select(s => s.Offset + s.Length).DefaultIfEmpty(ModelPackage.HeaderSize).Max();
        long modelEnd = end;
        foreach (var s in sections.Where(s => s.Offset + s.Length > modelEnd))
        {
            end = PackageWriter.Align(end, PackageWriter.SectionAlignment) + s.Length;
        }

        long table = PackageWriter.Align(end, PackageWriter.SectionAlignment);
        return table + sections.Sum(s => SectionTableEntryBytes + Encoding.UTF8.GetByteCount(s.Name));
    }
}
