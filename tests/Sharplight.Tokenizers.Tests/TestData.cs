using System.Text.Json;
using Sharplight.Tokenizers;

namespace Sharplight.Tokenizers.Tests;

internal static class TestData
{
    public static string? FindWorkFile(string relative)
    {
        string? env = Environment.GetEnvironmentVariable("SHARPLIGHT_TOKENIZER_DIR");
        if (env != null && File.Exists(Path.Combine(env, relative))) return Path.Combine(env, relative);
        for (var d = new DirectoryInfo(AppContext.BaseDirectory); d != null; d = d.Parent)
        {
            string p = Path.Combine(d.FullName, ".work", "tokenizer", relative);
            if (File.Exists(p)) return p;
        }
        return null;
    }

    private static TokenizerData? _qwen;
    private static bool _qwenLoaded;

    /// <summary>The Qwen3.8 tokenizer exported from the GGUF by .work/tokenizer/export_gguf_tokenizer.py; null if not exported.</summary>
    public static TokenizerData? Qwen35()
    {
        if (_qwenLoaded) return _qwen;
        _qwenLoaded = true;
        string? path = FindWorkFile("qwen35.json");
        if (path == null) return null;
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        var md = new Dictionary<string, object>();
        foreach (var p in doc.RootElement.EnumerateObject())
        {
            // Fixture is a GGUF metadata dump; translate to canonical keys like the GGUF importer does.
            md[p.Name.Replace("tokenizer.ggml.", "tokenizer.")] = p.Value.ValueKind switch
            {
                JsonValueKind.String => p.Value.GetString()!,
                JsonValueKind.True => true,
                JsonValueKind.False => false,
                JsonValueKind.Number => p.Value.GetInt64(),
                JsonValueKind.Array => p.Value.EnumerateArray().Select(e => e.ValueKind == JsonValueKind.String ? (object)e.GetString()! : e.GetInt64()).ToArray(),
                _ => throw new InvalidDataException(p.Name),
            };
        }
        return _qwen = TokenizerData.FromMetadata(md);
    }

    private static ITokenizer? _qwenTok;
    public static ITokenizer? QwenTokenizer() => _qwenTok ??= Qwen35() is { } d ? TokenizerFactory.Create(d) : null;

    public sealed record Case(string Text, bool ParseSpecial, int[] Ids);

    public static List<Case> LoadCases(string path)
    {
        using var doc = JsonDocument.Parse(File.ReadAllBytes(path));
        return doc.RootElement.EnumerateArray().Select(e => new Case(
            e.GetProperty("text").GetString()!,
            e.GetProperty("parseSpecial").GetBoolean(),
            e.GetProperty("ids").EnumerateArray().Select(i => i.GetInt32()).ToArray())).ToList();
    }

    private static List<Case>? _fixture;
    public static List<Case> Fixture() => _fixture ??= LoadCases(Path.Combine(AppContext.BaseDirectory, "Fixtures", "qwen35_cases.json"));

    public static IEnumerable<object[]> FixtureIndexes() => Enumerable.Range(0, Fixture().Count).Select(i => new object[] { i });

    /// <summary>Tiny hand-made byte-level BPE: 256 byte tokens, a few merges, a control token and a user-defined token.</summary>
    public static TokenizerData Tiny(string pre = "default", bool addBos = false)
    {
        var tokens = new List<string>();
        var types = new List<int>();
        for (int b = 0; b < 256; b++) { tokens.Add(Sharplight.Tokenizers.Bpe.ByteEncoding.ByteToChar[b].ToString()); types.Add(1); }
        string G = Sharplight.Tokenizers.Bpe.ByteEncoding.ByteToChar[32].ToString();
        string[] merges = ["h e", "l l", "he ll", "hell o", G + " w", "o r", G + "w or", "l d", G + "wor ld"];
        foreach (var m in merges)
        {
            var parts = m.Split(' ', 2);
            // note: merge strings for tokens that start with the space marker use that marker as first char
            tokens.Add(parts[0] + parts[1]); types.Add(1);
        }
        tokens.Add("<s>"); types.Add(3);
        tokens.Add("</s>"); types.Add(3);
        tokens.Add("<usr>"); types.Add(4);
        // fix up: merges must reference existing symbols; add intermediate tokens as needed (done by virtual symbols otherwise)
        return new TokenizerData
        {
            Model = "gpt2", PreTokenizer = pre, Tokens = tokens.ToArray(), TokenTypes = types.ToArray(),
            Merges = merges, BosId = tokens.Count - 3, EosId = tokens.Count - 2, EosIds = [tokens.Count - 2], AddBos = addBos,
        };
    }
}
