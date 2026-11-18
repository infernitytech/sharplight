using System.Security.Cryptography;
using System.Text;

namespace Sharplight.Studio.ViewModels.Search;

/// <summary>
/// The index of a folder as a compact binary file in the page's directory: header (model, passage settings), the file fingerprints, then
/// every passage with its position and its embedding as 16-bit floats. Written to a temporary file and moved into place, so an
/// interrupted write never damages the last good index.
/// </summary>
internal static class SearchIndexStore
{
    private const int Magic = 0x58534C53;   // "SLSX"
    private const int Version = 1;

    /// <summary>Where the index of <paramref name="folder"/> lives in <paramref name="directory"/>.</summary>
    /// <param name="directory">The page's directory.</param>
    /// <param name="folder">The indexed folder.</param>
    public static string PathFor(string directory, string folder)
    {
        string full = Path.TrimEndingDirectorySeparator(Path.GetFullPath(folder));
        string name = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(full)))[..16].ToLowerInvariant();
        return Path.Combine(directory, "indexes", name + ".slsx");
    }

    /// <summary>Writes <paramref name="index"/> to <paramref name="path"/>.</summary>
    /// <param name="path">The index file.</param>
    /// <param name="index">The index.</param>
    public static void Save(string path, SearchIndex index)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string temporary = path + ".tmp";
        using (var stream = new FileStream(temporary, FileMode.Create, FileAccess.Write, FileShare.None))
        using (var writer = new BinaryWriter(stream, Encoding.UTF8))
        {
            int dimensions = index.Chunks.Count == 0 ? 0 : index.Chunks[0].Vector.Length;
            writer.Write(Magic);
            writer.Write(Version);
            writer.Write(index.Folder);
            writer.Write(index.ModelId);
            writer.Write(index.ChunkCharacters);
            writer.Write(index.OverlapCharacters);
            writer.Write(dimensions);
            writer.Write(index.Files.Count);
            var positions = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var file in index.Files)
            {
                positions[file.Path] = positions.Count;
                writer.Write(file.Path);
                writer.Write(file.Size);
                writer.Write(file.ModifiedTicks);
                writer.Write(file.Hash);
                writer.Write(file.Error);
            }

            writer.Write(index.Chunks.Count);
            foreach (var chunk in index.Chunks)
            {
                string relative = Path.GetRelativePath(index.Folder, chunk.File).Replace('\\', '/');
                writer.Write(positions.GetValueOrDefault(relative, -1));
                writer.Write(chunk.Start);
                writer.Write(chunk.Page);
                writer.Write(chunk.Line);
                writer.Write(chunk.Heading);
                writer.Write(chunk.Text);
                for (int i = 0; i < dimensions; i++)
                {
                    writer.Write(i < chunk.Vector.Length ? (Half)chunk.Vector[i] : (Half)0);
                }
            }
        }

        File.Move(temporary, path, overwrite: true);
    }

    /// <summary>Reads an index; null when the file is missing, damaged or of another version.</summary>
    /// <param name="path">The index file.</param>
    public static SearchIndex? Load(string path)
    {
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            using var reader = new BinaryReader(stream, Encoding.UTF8);
            if (reader.ReadInt32() != Magic || reader.ReadInt32() != Version)
            {
                return null;
            }

            string folder = reader.ReadString();
            string model = reader.ReadString();
            int chunkCharacters = reader.ReadInt32();
            int overlapCharacters = reader.ReadInt32();
            int dimensions = reader.ReadInt32();
            int fileCount = reader.ReadInt32();
            var files = new List<IndexedFile>(fileCount);
            for (int i = 0; i < fileCount; i++)
            {
                files.Add(new IndexedFile(reader.ReadString(), reader.ReadInt64(), reader.ReadInt64(), reader.ReadString(), reader.ReadString()));
            }

            int chunkCount = reader.ReadInt32();
            var chunks = new List<IndexedChunk>(chunkCount);
            for (int c = 0; c < chunkCount; c++)
            {
                int file = reader.ReadInt32();
                int start = reader.ReadInt32();
                int page = reader.ReadInt32();
                int line = reader.ReadInt32();
                string heading = reader.ReadString();
                string text = reader.ReadString();
                var vector = new float[dimensions];
                for (int i = 0; i < dimensions; i++)
                {
                    vector[i] = (float)reader.ReadHalf();
                }

                if (file < 0 || file >= files.Count)
                {
                    return null;
                }

                chunks.Add(new IndexedChunk(Path.Combine(folder, files[file].Path.Replace('/', Path.DirectorySeparatorChar)), start, text, SearchIndex.Normalize(vector), page, line, heading));
            }

            return new SearchIndex(folder, model, files.Count(file => file.Error.Length == 0), chunks)
            {
                Files = files,
                ChunkCharacters = chunkCharacters,
                OverlapCharacters = overlapCharacters,
            };
        }
        catch (Exception failure) when (failure is IOException or EndOfStreamException or InvalidDataException or FormatException or UnauthorizedAccessException or ArgumentException)
        {
            return null;
        }
    }
}
