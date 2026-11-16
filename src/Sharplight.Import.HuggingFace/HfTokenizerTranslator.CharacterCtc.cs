using System.Text.Json;

namespace Sharplight.Import.HuggingFace;

// Character-level CTC tokenizers saved as vocab.json only (Wav2Vec2CTCTokenizer: wav2vec2, HuBERT, XLSR fine-tunes).
internal static partial class HfTokenizerTranslator
{
    private const string CtcWordDelimiter = "|";

    private static readonly string[] s_ctcSpecialTokens = ["<pad>", "<s>", "</s>", "<unk>", "[PAD]", "[UNK]", "[CLS]", "[SEP]"];

    /// <summary>Whether <paramref name="dir"/> holds a character-level CTC tokenizer (vocab.json, no merges, no tokenizer.json) of a speech model.</summary>
    /// <param name="dir">The checkpoint directory.</param>
    /// <param name="config">The checkpoint's configuration.</param>
    private static bool IsCharacterCtc(string dir, HfConfig config)
    {
        if (!File.Exists(Path.Combine(dir, SlowVocabularyFile)) || File.Exists(Path.Combine(dir, SlowMergesFile)))
        {
            return false;
        }

        if (config.ModelType is "wav2vec2" or "hubert" or "wav2vec2-conformer" or "data2vec-audio" or "unispeech" or "unispeech-sat" or "wavlm" or "sew" or "sew-d")
        {
            return true;
        }

        var tokenizerConfig = ReadTokenizerConfig(dir);
        return tokenizerConfig is { } c && c.TryGetProperty("tokenizer_class", out var kind) && kind.GetString() is { } name && name.Contains("CTC", StringComparison.Ordinal);
    }

    /// <summary>
    /// A Unigram <c>tokenizer.json</c> of the characters in vocabulary order: the word delimiter <c>|</c> becomes the SentencePiece space
    /// <c>▁</c> (a CTC transcript is its pieces joined, so the delimiters turn into the spaces between words), the special tokens are added
    /// tokens. Matches <c>Wav2Vec2CTCTokenizer.decode</c>.
    /// </summary>
    internal static byte[] ConvertCharacterCtc(string dir)
    {
        using var document = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(dir, SlowVocabularyFile)));
        var entries = document.RootElement.EnumerateObject().Select(p => (Piece: p.Name, Id: p.Value.GetInt32())).OrderBy(e => e.Id).ToList();
        var tokenizerConfig = ReadTokenizerConfig(dir);
        string delimiter = tokenizerConfig is { } c && c.TryGetProperty("word_delimiter_token", out var d) && d.ValueKind == JsonValueKind.String ? d.GetString()! : CtcWordDelimiter;
        int unk = entries.FindIndex(e => e.Piece is "<unk>" or "[UNK]");

        using var stream = new MemoryStream();
        using (var w = new Utf8JsonWriter(stream))
        {
            w.WriteStartObject();
            w.WriteStartArray("added_tokens");
            foreach (var (piece, id) in entries.Where(e => Array.IndexOf(s_ctcSpecialTokens, e.Piece) >= 0))
            {
                w.WriteStartObject();
                w.WriteNumber("id", id);
                w.WriteString("content", piece);
                w.WriteBoolean("special", true);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteStartObject("model");
            w.WriteString("type", "Unigram");
            if (unk >= 0)
            {
                w.WriteNumber("unk_id", entries[unk].Id);
            }

            w.WriteStartArray("vocab");
            int next = 0;
            foreach (var (piece, id) in entries)
            {
                for (; next < id; next++)
                {
                    WritePiece(w, $"[UNUSED{next}]");
                }

                WritePiece(w, piece == delimiter ? "▁" : piece);
                next = id + 1;
            }

            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteEndObject();
        }

        return stream.ToArray();

        static void WritePiece(Utf8JsonWriter w, string piece)
        {
            w.WriteStartArray();
            w.WriteStringValue(piece);
            w.WriteNumberValue(0.0);
            w.WriteEndArray();
        }
    }
}
