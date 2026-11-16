using System.Text;
using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

// "Slow" byte-level BPE tokenizers saved without tokenizer.json (GPT2Tokenizer / RobertaTokenizer style: vocab.json + merges.txt).
internal static partial class HfTokenizerTranslator
{
    private const string SlowVocabularyFile = "vocab.json";
    private const string SlowMergesFile = "merges.txt";

    /// <summary>The special-token keys of tokenizer_config.json that become added tokens of the converted tokenizer.</summary>
    private static readonly string[] s_slowSpecialKeys = ["bos_token", "eos_token", "unk_token", "pad_token", "sep_token", "cls_token", "mask_token"];

    /// <summary>Whether <paramref name="dir"/> holds a slow byte-level BPE tokenizer (vocab.json and merges.txt) instead of tokenizer.json.</summary>
    private static bool IsSlowByteLevelBpe(string dir)
        => File.Exists(Path.Combine(dir, SlowVocabularyFile)) && File.Exists(Path.Combine(dir, SlowMergesFile));

    /// <summary>
    /// Builds the tokenizer.json that transformers' <c>GPT2Converter</c> makes from <c>vocab.json</c>, <c>merges.txt</c> and
    /// <c>tokenizer_config.json</c>: a byte-level BPE with the GPT-2 split, the configured <c>add_prefix_space</c>, the special
    /// tokens as added tokens, and a BOS-prepending template when <c>add_bos_token</c> is set (OPT); otherwise no tokens are added.
    /// </summary>
    internal static byte[] ConvertSlowByteLevelBpe(string dir)
    {
        using var vocabDocument = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, SlowVocabularyFile)));
        var vocab = vocabDocument.RootElement;
        var tokenizerConfig = ReadTokenizerConfig(dir);
        bool addPrefixSpace = tokenizerConfig is { } c1 && c1.TryGetProperty("add_prefix_space", out var aps) && aps.ValueKind == JsonValueKind.True;
        bool addBos = tokenizerConfig is { } c2 && c2.TryGetProperty("add_bos_token", out var ab) && ab.ValueKind == JsonValueKind.True;

        using var stream = new MemoryStream();
        using (var w = new Utf8JsonWriter(stream))
        {
            w.WriteStartObject();
            w.WriteStartArray("added_tokens");
            var written = new HashSet<string>(StringComparer.Ordinal);
            string? bos = null;
            int bosId = -1;
            foreach (string key in s_slowSpecialKeys)
            {
                if (SpecialTokenContent(tokenizerConfig, key) is not { } content || !vocab.TryGetProperty(content, out var idElement))
                {
                    continue;
                }

                if (key == "bos_token")
                {
                    bos = content;
                    bosId = idElement.GetInt32();
                }

                if (!written.Add(content))
                {
                    continue;
                }

                w.WriteStartObject();
                w.WriteNumber("id", idElement.GetInt32());
                w.WriteString("content", content);
                w.WriteBoolean("single_word", false);
                w.WriteBoolean("lstrip", false);
                w.WriteBoolean("rstrip", false);
                w.WriteBoolean("normalized", true);
                w.WriteBoolean("special", true);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteStartObject("pre_tokenizer");
            w.WriteString("type", "ByteLevel");
            w.WriteBoolean("add_prefix_space", addPrefixSpace);
            w.WriteBoolean("trim_offsets", true);
            w.WriteBoolean("use_regex", true);
            w.WriteEndObject();
            w.WriteStartObject("post_processor");
            if (addBos && bos is not null)
            {
                w.WriteString("type", "TemplateProcessing");
                w.WriteStartArray("single");
                w.WriteStartObject();
                w.WriteStartObject("SpecialToken");
                w.WriteString("id", bos);
                w.WriteNumber("type_id", 0);
                w.WriteEndObject();
                w.WriteEndObject();
                w.WriteStartObject();
                w.WriteStartObject("Sequence");
                w.WriteString("id", "A");
                w.WriteNumber("type_id", 0);
                w.WriteEndObject();
                w.WriteEndObject();
                w.WriteEndArray();
                w.WriteStartObject("special_tokens");
                w.WriteStartObject(bos);
                w.WriteString("id", bos);
                w.WriteStartArray("ids");
                w.WriteNumberValue(bosId);
                w.WriteEndArray();
                w.WriteStartArray("tokens");
                w.WriteStringValue(bos);
                w.WriteEndArray();
                w.WriteEndObject();
                w.WriteEndObject();
            }
            else
            {
                w.WriteString("type", "ByteLevel");
                w.WriteBoolean("trim_offsets", false);
            }

            w.WriteEndObject();
            w.WriteStartObject("decoder");
            w.WriteString("type", "ByteLevel");
            w.WriteEndObject();
            w.WriteStartObject("model");
            w.WriteString("type", "BPE");
            w.WritePropertyName("vocab");
            vocab.WriteTo(w);
            w.WriteStartArray("merges");
            foreach (string line in File.ReadLines(Path.Combine(dir, SlowMergesFile), Encoding.UTF8))
            {
                if (line.StartsWith("#version", StringComparison.Ordinal) || line.Length == 0)
                {
                    continue;
                }

                w.WriteStringValue(line);
            }

            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndObject();
        }

        return stream.ToArray();
    }

    /// <summary>The text of a special token in tokenizer_config.json (a string or an AddedToken object), or null.</summary>
    private static string? SpecialTokenContent(JsonElement? tokenizerConfig, string key)
    {
        if (tokenizerConfig is not { } config || !config.TryGetProperty(key, out var value))
        {
            return null;
        }

        return value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            JsonValueKind.Object when value.TryGetProperty("content", out var content) => content.GetString(),
            _ => null,
        };
    }
}
