using System.Text;

namespace Sharplight.Import.PyTorch;

/// <summary>
/// Reads the central directory of a zip archive (including zip64) that stores its entries uncompressed, as <c>torch.save</c> does,
/// so that entry data can be addressed in place. The directory is the only source of truth: local headers are used solely to find
/// where an entry's data starts, and every offset and size is validated against the file length.
/// </summary>
internal sealed class ZipDirectory
{
    private const uint EndOfCentralDirectory = 0x06054b50;
    private const uint Zip64Locator = 0x07064b50;
    private const uint Zip64EndOfCentralDirectory = 0x06064b50;
    private const uint CentralHeader = 0x02014b50;
    private const uint LocalHeader = 0x04034b50;
    private const long MaxEntries = 4_000_000;

    private readonly Dictionary<string, ZipEntry> _entries = new(StringComparer.Ordinal);

    private ZipDirectory()
    {
    }

    /// <summary>The entries by name.</summary>
    public IReadOnlyDictionary<string, ZipEntry> Entries => _entries;

    /// <summary>True when <paramref name="file"/> starts like a zip archive.</summary>
    public static bool LooksLikeZip(FileView file) => file.Length >= 4 && file.U32(0) is LocalHeader or EndOfCentralDirectory;

    /// <summary>Parses the central directory of <paramref name="file"/>.</summary>
    /// <exception cref="InvalidDataException">The archive is malformed, has duplicate names, or compresses an entry.</exception>
    public static ZipDirectory Read(FileView file)
    {
        long eocd = FindEndOfCentralDirectory(file);
        long count = file.U16(eocd + 10);
        long directorySize = file.U32(eocd + 12);
        long directoryOffset = file.U32(eocd + 16);
        if ((count == 0xFFFF || directorySize == 0xFFFFFFFF || directoryOffset == 0xFFFFFFFF) && eocd >= 20 && file.U32(eocd - 20) == Zip64Locator)
        {
            long record = file.I64(eocd - 20 + 8);
            if (file.U32(record) != Zip64EndOfCentralDirectory)
            {
                throw new InvalidDataException("Invalid zip64 end of central directory.");
            }

            count = file.I64(record + 32);
            directorySize = file.I64(record + 40);
            directoryOffset = file.I64(record + 48);
        }

        if (count > MaxEntries || directoryOffset + directorySize > file.Length)
        {
            throw new InvalidDataException("The zip central directory lies outside the file.");
        }

        var zip = new ZipDirectory();
        long position = directoryOffset;
        long end = directoryOffset + directorySize;
        for (long i = 0; i < count; i++)
        {
            if (position + 46 > end || file.U32(position) != CentralHeader)
            {
                throw new InvalidDataException("Invalid zip central directory entry.");
            }

            ushort method = file.U16(position + 10);
            long compressedSize = file.U32(position + 20);
            long size = file.U32(position + 24);
            int nameLength = file.U16(position + 28);
            int extraLength = file.U16(position + 30);
            int commentLength = file.U16(position + 32);
            long localOffset = file.U32(position + 42);
            long next = position + 46 + nameLength + extraLength + commentLength;
            if (next > end)
            {
                throw new InvalidDataException("A zip central directory entry runs past the directory.");
            }

            string name = Encoding.UTF8.GetString(file.Slice(position + 46, nameLength));
            ApplyZip64Extra(file.Slice(position + 46 + nameLength, extraLength), ref size, ref compressedSize, ref localOffset);
            position = next;
            if (name.EndsWith('/'))
            {
                continue;
            }

            if (method != 0 || compressedSize != size)
            {
                throw new InvalidDataException($"Zip entry '{name}' is compressed; PyTorch checkpoints store tensors uncompressed.");
            }

            long dataOffset = LocateData(file, localOffset, size, name);
            if (!zip._entries.TryAdd(name, new ZipEntry(name, dataOffset, size)))
            {
                throw new InvalidDataException($"Duplicate zip entry '{name}'.");
            }
        }

        return zip;
    }

    private static long LocateData(FileView file, long localOffset, long size, string name)
    {
        if (file.U32(localOffset) != LocalHeader)
        {
            throw new InvalidDataException($"Zip entry '{name}' has an invalid local header.");
        }

        int nameLength = file.U16(localOffset + 26);
        int extraLength = file.U16(localOffset + 28);
        long data = localOffset + 30 + nameLength + extraLength;
        if (size < 0 || data + size > file.Length)
        {
            throw new InvalidDataException($"Zip entry '{name}' lies outside the file.");
        }

        return data;
    }

    private static void ApplyZip64Extra(ReadOnlySpan<byte> extra, ref long size, ref long compressedSize, ref long localOffset)
    {
        while (extra.Length >= 4)
        {
            ushort id = System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(extra);
            int length = System.Buffers.Binary.BinaryPrimitives.ReadUInt16LittleEndian(extra[2..]);
            if (4 + length > extra.Length)
            {
                break;
            }

            if (id == 1)
            {
                var field = extra.Slice(4, length);
                int at = 0;
                ReadWide(field, ref at, ref size);
                ReadWide(field, ref at, ref compressedSize);
                ReadWide(field, ref at, ref localOffset);
            }

            extra = extra[(4 + length)..];
        }
    }

    private static void ReadWide(ReadOnlySpan<byte> field, ref int at, ref long value)
    {
        if (value != 0xFFFFFFFF || at + 8 > field.Length)
        {
            return;
        }

        long wide = System.Buffers.Binary.BinaryPrimitives.ReadInt64LittleEndian(field[at..]);
        value = wide >= 0 ? wide : throw new InvalidDataException("A zip64 field is out of range.");
        at += 8;
    }

    private static long FindEndOfCentralDirectory(FileView file)
    {
        long lowest = Math.Max(0, file.Length - 22 - ushort.MaxValue);
        for (long i = file.Length - 22; i >= lowest; i--)
        {
            if (file.U32(i) == EndOfCentralDirectory)
            {
                return i;
            }
        }

        throw new InvalidDataException("No zip end of central directory found.");
    }
}
