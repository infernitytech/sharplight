using System.IO.Compression;

namespace Sharplight.Import.PyTorch.Tests;

/// <summary>Writes checkpoint files for tests.</summary>
internal static class CheckpointFiles
{
    public static string NewDirectory()
    {
        string dir = Path.Combine(Path.GetTempPath(), "sl-pytorch-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        return dir;
    }

    /// <summary>A torch-style zip archive with uncompressed entries under an <c>archive/</c> prefix.</summary>
    public static string WriteZip(string dir, byte[] pickle, params (string Key, byte[] Data)[] storages)
    {
        string path = Path.Combine(dir, Guid.NewGuid().ToString("N") + ".pt");
        using var stream = File.Create(path);
        using var zip = new ZipArchive(stream, ZipArchiveMode.Create);
        Add(zip, "archive/data.pkl", pickle);
        Add(zip, "archive/byteorder", "little"u8.ToArray());
        foreach (var (key, data) in storages)
        {
            Add(zip, "archive/data/" + key, data);
        }

        Add(zip, "archive/version", "3\n"u8.ToArray());
        zip.Dispose();
        return path;
    }

    public static string WriteBytes(string dir, byte[] content)
    {
        string path = Path.Combine(dir, Guid.NewGuid().ToString("N") + ".bin");
        File.WriteAllBytes(path, content);
        return path;
    }

    private static void Add(ZipArchive zip, string name, byte[] data)
    {
        var entry = zip.CreateEntry(name, CompressionLevel.NoCompression);
        using var stream = entry.Open();
        stream.Write(data);
    }

    public static byte[] Floats(params float[] values)
    {
        var bytes = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return bytes;
    }
}
