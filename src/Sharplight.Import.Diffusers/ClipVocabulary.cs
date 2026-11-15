using System.IO.Compression;
using System.Text.Json;
using Sharplight.Models;

namespace Sharplight.Import.Diffusers;

/// <summary>
/// The CLIP byte-level BPE vocabulary of a text encoder, written as the canonical tokenizer keys (<c>tokenizer.model = "clip"</c>):
/// read from a tokenizer folder (<c>vocab.json</c> + <c>merges.txt</c>, or <c>tokenizer.json</c>), or rebuilt from the embedded
/// OpenAI CLIP merges for checkpoints that carry no tokenizer (single-file and GGUF), the way OpenAI CLIP builds it: the 256 byte
/// symbols, the same with the end-of-word marker, one token per merge, then the start and end tokens.
/// </summary>
internal sealed class ClipVocabulary
{
    private const string StartToken = "<|startoftext|>";
    private const string EndToken = "<|endoftext|>";
    private const string MergesResource = "Sharplight.Import.Diffusers.clip-merges.txt.gz";

    private ClipVocabulary(string[] tokens, string[] merges, string padToken)
    {
        Tokens = tokens;
        Merges = merges;
        PadToken = padToken;
    }

    /// <summary>Token strings by id (end-of-word symbols carry <c>&lt;/w&gt;</c>).</summary>
    public string[] Tokens { get; }

    /// <summary>Merges as "left right", by rank.</summary>
    public string[] Merges { get; }

    /// <summary>The padding token (the end token for OpenAI CLIP, <c>!</c> for the OpenCLIP encoders of SD 2.x and SDXL).</summary>
    public string PadToken { get; }

    /// <summary>Reads a tokenizer folder of a diffusers pipeline.</summary>
    /// <param name="folder">The folder (<c>tokenizer/</c>).</param>
    /// <exception cref="InvalidDataException">The folder has neither vocab.json + merges.txt nor tokenizer.json.</exception>
    public static ClipVocabulary FromFolder(string folder)
    {
        string pad = PadFromConfig(folder);
        string vocabPath = Path.Combine(folder, "vocab.json");
        string mergesPath = Path.Combine(folder, "merges.txt");
        if (File.Exists(vocabPath) && File.Exists(mergesPath))
        {
            using var vocab = JsonDocument.Parse(File.ReadAllBytes(vocabPath));
            var merges = File.ReadLines(mergesPath).Where(l => l.Length > 0 && !l.StartsWith("#version", StringComparison.Ordinal)).ToArray();
            return new ClipVocabulary(ToTokenArray(vocab.RootElement), merges, pad);
        }

        string jsonPath = Path.Combine(folder, "tokenizer.json");
        if (!File.Exists(jsonPath))
        {
            throw new InvalidDataException($"'{folder}' has no CLIP tokenizer (vocab.json + merges.txt or tokenizer.json).");
        }

        using var document = JsonDocument.Parse(File.ReadAllBytes(jsonPath));
        var model = document.RootElement.GetProperty("model");
        var mergeList = model.GetProperty("merges").EnumerateArray()
            .Select(m => m.ValueKind == JsonValueKind.Array ? $"{m[0].GetString()} {m[1].GetString()}" : m.GetString()!).ToArray();
        return new ClipVocabulary(ToTokenArray(model.GetProperty("vocab")), mergeList, pad);
    }

    /// <summary>The OpenAI CLIP vocabulary (SD 1.x), rebuilt from the embedded merges.</summary>
    /// <param name="padToken">The padding token.</param>
    public static ClipVocabulary OpenAi(string padToken = EndToken)
    {
        using var stream = typeof(ClipVocabulary).Assembly.GetManifestResourceStream(MergesResource)
            ?? throw new InvalidOperationException($"Missing embedded resource {MergesResource}.");
        using var gzip = new GZipStream(stream, CompressionMode.Decompress);
        using var reader = new StreamReader(gzip);
        var merges = new List<string>();
        while (reader.ReadLine() is { } line)
        {
            if (line.Length > 0)
            {
                merges.Add(line);
            }
        }

        var byteSymbols = ByteSymbols();
        var tokens = new List<string>(byteSymbols.Count * 2 + merges.Count + 2);
        tokens.AddRange(byteSymbols);
        tokens.AddRange(byteSymbols.Select(s => s + "</w>"));
        tokens.AddRange(merges.Select(m => m.Replace(" ", "", StringComparison.Ordinal)));
        tokens.Add(StartToken);
        tokens.Add(EndToken);
        return new ClipVocabulary([.. tokens], [.. merges], padToken);
    }

    /// <summary>Writes the canonical tokenizer keys (under <paramref name="prefix"/>, empty for the model's main tokenizer).</summary>
    /// <param name="metadata">The canonical metadata.</param>
    /// <param name="prefix">Key prefix, e.g. <c>text-encoder-2.</c>, or empty.</param>
    /// <param name="maxLength">Tokens per sequence including the start and end tokens.</param>
    public void WriteTo(IDictionary<string, object> metadata, string prefix, int maxLength)
    {
        int Id(string token) => Array.IndexOf(Tokens, token);
        metadata[prefix + CanonicalKeys.TokenizerModel] = "clip";
        metadata[prefix + CanonicalKeys.TokenizerTokens] = Tokens;
        metadata[prefix + CanonicalKeys.TokenizerMerges] = Merges;
        metadata[prefix + CanonicalKeys.TokenizerBosId] = Id(StartToken);
        metadata[prefix + CanonicalKeys.TokenizerEosId] = Id(EndToken);
        metadata[prefix + CanonicalKeys.TokenizerPadId] = Math.Max(0, Id(PadToken));
        metadata[prefix + CanonicalKeys.TokenizerMaxLength] = maxLength;
    }

    private static string[] ToTokenArray(JsonElement vocab)
    {
        var pairs = vocab.EnumerateObject().Select(p => (Token: p.Name, Id: p.Value.GetInt32())).ToList();
        var tokens = new string[pairs.Max(p => p.Id) + 1];
        foreach (var (token, id) in pairs)
        {
            tokens[id] = token;
        }

        for (int i = 0; i < tokens.Length; i++)
        {
            tokens[i] ??= $"<unused{i}>";
        }

        return tokens;
    }

    private static string PadFromConfig(string folder)
    {
        foreach (string file in new[] { "special_tokens_map.json", "tokenizer_config.json" })
        {
            string path = Path.Combine(folder, file);
            if (!File.Exists(path))
            {
                continue;
            }

            using var document = JsonDocument.Parse(File.ReadAllBytes(path));
            if (document.RootElement.TryGetProperty("pad_token", out var pad))
            {
                return pad.ValueKind == JsonValueKind.Object ? pad.GetProperty("content").GetString()! : pad.GetString() ?? EndToken;
            }
        }

        return EndToken;
    }

    /// <summary>GPT-2's printable stand-ins for the 256 byte values, in the order CLIP's vocabulary lists them.</summary>
    private static List<string> ByteSymbols()
    {
        var printable = new List<int>();
        printable.AddRange(Enumerable.Range('!', '~' - '!' + 1));
        printable.AddRange(Enumerable.Range('¡', '¬' - '¡' + 1));
        printable.AddRange(Enumerable.Range('®', 'ÿ' - '®' + 1));
        var symbols = printable.Select(b => ((char)b).ToString()).ToList();
        int next = 0;
        for (int b = 0; b < 256; b++)
        {
            if (!printable.Contains(b))
            {
                symbols.Add(((char)(256 + next)).ToString());
                next++;
            }
        }

        return symbols;
    }
}
