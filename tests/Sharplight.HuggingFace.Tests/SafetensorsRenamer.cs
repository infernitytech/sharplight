using System.Text.Json;
using System.Text.Json.Nodes;

namespace Sharplight.HuggingFace.Tests;

/// <summary>
/// Copies a single-file safetensors checkpoint directory with its tensors renamed (tests only): checks that a mapping accepts the
/// tensor names of a released checkpoint that a tiny fixture spells differently.
/// </summary>
internal static class SafetensorsRenamer
{
    /// <summary>Writes <paramref name="target"/>: the small files of <paramref name="source"/> and its tensors under <paramref name="rename"/>'s names, plus <paramref name="extra"/>.</summary>
    public static void Copy(string source, string target, Func<string, string> rename, SafetensorsWriter? extra = null)
    {
        Directory.CreateDirectory(target);
        foreach (string file in Directory.EnumerateFiles(source).Where(f => !f.EndsWith(".safetensors", StringComparison.Ordinal)))
        {
            File.Copy(file, Path.Combine(target, Path.GetFileName(file)));
        }

        byte[] bytes = File.ReadAllBytes(Path.Combine(source, "model.safetensors"));
        int headerLength = checked((int)BitConverter.ToUInt64(bytes, 0));
        var header = JsonNode.Parse(bytes.AsSpan(8, headerLength))!.AsObject();
        var renamed = new JsonObject();
        foreach (var (name, node) in header)
        {
            renamed[name == "__metadata__" ? name : rename(name)] = node!.DeepClone();
        }

        byte[] json = JsonSerializer.SerializeToUtf8Bytes(renamed);
        using (var stream = File.Create(Path.Combine(target, "model.safetensors")))
        {
            stream.Write(BitConverter.GetBytes((ulong)json.Length));
            stream.Write(json);
            stream.Write(bytes.AsSpan(8 + headerLength));
        }

        extra?.Write(Path.Combine(target, "model-extra.safetensors"));
    }
}
