using System.Text.Json;

namespace Sharplight.Import.Kokoro;

/// <summary>The <c>config.json</c> of a Kokoro checkpoint.</summary>
internal sealed class KokoroConfig
{
    private readonly JsonElement _root;

    private KokoroConfig(JsonElement root)
    {
        _root = root;
    }

    /// <summary>Whether <paramref name="path"/> is a <c>config.json</c> with the <c>istftnet</c> and <c>plbert</c> sections.</summary>
    /// <param name="path">The file.</param>
    public static bool Matches(string path)
    {
        if (!File.Exists(path))
        {
            return false;
        }

        try
        {
            using var document = JsonDocument.Parse(File.ReadAllBytes(path));
            return document.RootElement.ValueKind == JsonValueKind.Object && document.RootElement.TryGetProperty("istftnet", out _)
                   && document.RootElement.TryGetProperty("plbert", out _) && document.RootElement.TryGetProperty("vocab", out _);
        }
        catch (JsonException)
        {
            return false;
        }
    }

    /// <summary>Reads <paramref name="path"/>.</summary>
    /// <param name="path">The <c>config.json</c>.</param>
    public static KokoroConfig Read(string path) => new(JsonDocument.Parse(File.ReadAllBytes(path)).RootElement.Clone());

    /// <summary>An integer property, optionally of a nested section.</summary>
    /// <param name="section">The section name, or <see langword="null"/> for the root.</param>
    /// <param name="name">The property name.</param>
    public int Int(string? section, string name) => Section(section).GetProperty(name).GetInt32();

    /// <summary>An array of integers.</summary>
    /// <param name="section">The section name.</param>
    /// <param name="name">The property name.</param>
    public long[] Ints(string section, string name) => [.. Section(section).GetProperty(name).EnumerateArray().Select(e => e.GetInt64())];

    /// <summary>The dilations of the residual blocks: one row per block.</summary>
    public long[][] Dilations() => [.. Section("istftnet").GetProperty("resblock_dilation_sizes").EnumerateArray().Select(r => r.EnumerateArray().Select(e => e.GetInt64()).ToArray())];

    /// <summary>The symbol vocabulary (phoneme character to token id) as JSON text.</summary>
    public string VocabularyJson => _root.GetProperty("vocab").GetRawText();

    private JsonElement Section(string? name) => name is null ? _root : _root.GetProperty(name);
}
