using System.Buffers.Binary;
using System.Text;

namespace Sharplight.Package;

/// <summary>Low-level writing of package sections, the section table and the header.</summary>
internal static class PackageWriter
{
    /// <summary>Alignment of every section start.</summary>
    public const long SectionAlignment = 16;

    /// <summary>Writes a section at the next aligned position of <paramref name="fs"/>.</summary>
    public static PackageSection WriteSection(FileStream fs, PackageSectionKind kind, string name, Action<BinaryWriter> body)
    {
        long start = Align(fs.Position, SectionAlignment);
        Pad(fs, start);
        using (var w = new BinaryWriter(fs, Encoding.UTF8, leaveOpen: true))
        {
            body(w);
            w.Flush();
        }
        return new PackageSection(kind, name, start, fs.Position - start);
    }

    /// <summary>
    /// Appends the section table at the end of <paramref name="fs"/>, makes it durable, then switches the header to it (written with
    /// <see cref="ModelPackage.Version"/>). Readers of the old table are unaffected: nothing they read is modified.
    /// </summary>
    public static void WriteTable(FileStream fs, IReadOnlyList<PackageSection> sections)
    {
        long table = Align(fs.Seek(0, SeekOrigin.End), SectionAlignment);
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
        Span<byte> h = stackalloc byte[ModelPackage.HeaderSize];
        ModelPackage.Magic.CopyTo(h);
        BinaryPrimitives.WriteInt32LittleEndian(h[8..], ModelPackage.Version);
        BinaryPrimitives.WriteInt32LittleEndian(h[12..], ModelPackage.HeaderSize);
        BinaryPrimitives.WriteInt64LittleEndian(h[16..], table);
        BinaryPrimitives.WriteInt32LittleEndian(h[24..], sections.Count);
        fs.Seek(0, SeekOrigin.Begin);
        fs.Write(h);
        fs.Flush(flushToDisk: true);
    }

    /// <summary>Writes zeros up to <paramref name="to"/>.</summary>
    /// <exception cref="InvalidOperationException">The stream is already past <paramref name="to"/>.</exception>
    public static void Pad(Stream fs, long to)
    {
        if (fs.Position > to)
        {
            throw new InvalidOperationException("Package writer position overran its layout.");
        }

        if (fs.Position < to)
        {
            fs.Write(new byte[to - fs.Position]);
        }
    }

    /// <summary>Rounds <paramref name="value"/> up to a multiple of <paramref name="alignment"/>.</summary>
    public static long Align(long value, long alignment) => (value + alignment - 1) / alignment * alignment;
}
