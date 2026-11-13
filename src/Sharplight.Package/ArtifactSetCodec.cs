using System.Text;
using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>
/// Encodes artifact sections. ARTS (version 2): <c>i32 format</c>, descriptor (<c>i32 count</c>, key/value strings), then the
/// entry directory and blobs. KSET (version 1, read only): entry directory and blobs. The directory is <c>i32 count</c>, then
/// per entry the key, <c>i64 offset</c> (relative to the section start) and <c>i64 length</c>; blobs are 16-byte aligned.
/// </summary>
internal static class ArtifactSetCodec
{
    private const int FormatVersion = 1;
    private const long BlobAlignment = 16;

    /// <summary>Appends <paramref name="set"/> as an ARTS section named by its descriptor id.</summary>
    public static PackageSection Write(FileStream fs, CompilationCache set)
    {
        var entries = set.Entries.OrderBy(e => e.Key, StringComparer.Ordinal).ToList();
        var descriptor = set.Descriptor;
        return PackageWriter.WriteSection(fs, PackageSectionKind.ArtifactSet, ArtifactDescriptor.Id(descriptor), w =>
        {
            // The header size does not depend on the offset values: lay it out once with zeros, then with the real offsets.
            var offsets = new long[entries.Count];
            var header = Header(descriptor, entries, offsets);
            long position = PackageWriter.Align(header.Length, BlobAlignment);
            for (int i = 0; i < entries.Count; i++)
            {
                offsets[i] = position;
                position = PackageWriter.Align(position + entries[i].Value.Length, BlobAlignment);
            }

            header = Header(descriptor, entries, offsets);
            w.Write(header);
            long written = header.Length;
            for (int i = 0; i < entries.Count; i++)
            {
                w.Write(new byte[offsets[i] - written]);
                w.Write(entries[i].Value.Span);
                written = offsets[i] + entries[i].Value.Length;
            }
        });
    }

    private static byte[] Header(IReadOnlyDictionary<string, string> descriptor, List<KeyValuePair<string, ReadOnlyMemory<byte>>> entries, long[] offsets)
    {
        var ms = new MemoryStream();
        using (var w = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(FormatVersion);
            w.Write(descriptor.Count);
            foreach (var key in descriptor.Keys.Order(StringComparer.Ordinal))
            {
                MetadataCodec.WriteString(w, key);
                MetadataCodec.WriteString(w, descriptor[key]);
            }

            w.Write(entries.Count);
            for (int i = 0; i < entries.Count; i++)
            {
                MetadataCodec.WriteString(w, entries[i].Key);
                w.Write(offsets[i]);
                w.Write((long)entries[i].Value.Length);
            }
        }
        return ms.ToArray();
    }

    /// <summary>Reads the descriptor and entry directory of an ARTS or KSET section; <paramref name="r"/> is positioned at its start.</summary>
    /// <exception cref="InvalidDataException">The section is malformed or of an unknown format version.</exception>
    public static ArtifactSetDirectory ReadDirectory(BinaryReader r, PackageSection section)
    {
        IReadOnlyDictionary<string, string> descriptor;
        if (section.Kind == PackageSectionKind.KernelSet)
        {
            descriptor = LegacyDescriptor(section.Name);
        }
        else
        {
            int format = r.ReadInt32();
            if (format != FormatVersion)
            {
                throw new InvalidDataException($"Artifact set '{section.Name}' has unsupported format {format}.");
            }

            int pairs = r.ReadInt32();
            var d = new Dictionary<string, string>(pairs, StringComparer.Ordinal);
            for (int i = 0; i < pairs; i++)
            {
                string key = MetadataCodec.ReadString(r);
                d[key] = MetadataCodec.ReadString(r);
            }
            descriptor = d;
        }

        int count = r.ReadInt32();
        if (count < 0)
        {
            throw new InvalidDataException($"Artifact set '{section.Name}' has a negative entry count.");
        }

        var entries = new List<(string, long, int)>(count);
        for (int i = 0; i < count; i++)
        {
            string key = MetadataCodec.ReadString(r);
            long offset = r.ReadInt64();
            long length = r.ReadInt64();
            if (offset < 0 || length < 0 || length > int.MaxValue || offset + length > section.Length)
            {
                throw new InvalidDataException($"Package entry {key} lies outside its section.");
            }

            entries.Add((key, section.Offset + offset, (int)length));
        }
        return new ArtifactSetDirectory(section, descriptor, entries);
    }

    /// <summary>
    /// Descriptor of a version 1 kernel set: tier <see cref="ArtifactTiers.Mixed"/>, the provider taken from the target string
    /// (<c>provider|device|driver|features</c>, see <see cref="IComputeDevice.CompilationTarget"/>), and the target itself.
    /// </summary>
    public static IReadOnlyDictionary<string, string> LegacyDescriptor(string target) => new Dictionary<string, string>(StringComparer.Ordinal)
    {
        [ArtifactDescriptorKeys.Provider] = target.Split('|')[0],
        [ArtifactDescriptorKeys.Tier] = ArtifactTiers.Mixed,
        [ArtifactDescriptorKeys.Kind] = ArtifactDescriptorKeys.LegacyKernelSetKind,
        [ArtifactDescriptorKeys.Target] = target,
    };
}
