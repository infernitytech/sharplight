using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tokenizers;

namespace Sharplight.Import.Gguf;

/// <summary>
/// GGUF <c>whisper</c> files as transcribe.cpp writes them (the self-describing layout of the popular <c>whisper-*-gguf</c> repositories:
/// <c>stt.whisper.*</c> hyper-parameters, <c>stt.frontend.*</c> log-mel settings, the filter bank as the tensor
/// <c>frontend.mel_filterbank</c>, the byte-level BPE vocabulary with its merges, <c>enc.*</c> / <c>dec.*</c> tensors) → canonical
/// <c>encoder_decoder</c> with an audio encoder, the same canonical model a Hugging Face or whisper.cpp Whisper checkpoint imports to.
/// The filter bank becomes the <c>feature-extractor/mel-filters</c> attachment; the decoding keys (<see cref="SpeechKeys"/>) come from the
/// stored special token ids and suppress lists, the languages from the language tokens in vocabulary order.
/// </summary>
/// <remarks>
/// Candle's whisper GGUF files (Hugging Face tensor names under <c>model.</c>, the tokenizer and configuration in separate JSON files)
/// share the architecture id but not the layout; <see cref="Validate"/> names them instead of failing on the first tensor.
/// </remarks>
internal sealed partial class WhisperGgufMapping : IGgufArchitectureMapping
{
    private const string Stt = "stt.whisper.";
    private const string Frontend = "stt.frontend.";
    private const string MelFilters = "feature-extractor/mel-filters";
    private const string Window = "feature-extractor/window";
    private const int MultilingualVocabulary = 51865;

    /// <summary>Why Candle's whisper GGUF files are rejected, and what to use instead.</summary>
    public const string CandleLayout = "This is a Candle whisper GGUF (Hugging Face tensor names, the tokenizer and configuration in separate files), "
        + "which is not supported; use the whisper.cpp ggml-*.bin files or the transcribe.cpp whisper-*.gguf files of the same model.";

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["whisper"];

    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IEnumerable<string> Validate(GgufMetadataView g)
    {
        if (g.Tensor("enc.conv.0.weight") is null)
        {
            if (g.Tensor("model.encoder.conv1.weight") is not null || g.Tensor("encoder.conv1.weight") is not null)
            {
                yield return CandleLayout;
            }
            else
            {
                yield return "The whisper GGUF has no encoder convolution 'enc.conv.0.weight' (not the transcribe.cpp layout).";
            }

            yield break;
        }

        if (Int(g, Stt + "encoder.d_model") != Int(g, Stt + "decoder.d_model") || Int(g, Stt + "encoder.n_heads") != Int(g, Stt + "decoder.n_heads"))
        {
            yield return "Encoder and decoder of different widths or head counts are not supported.";
        }

        foreach (string part in new[] { "encoder", "decoder" })
        {
            if (Text(g, Stt + part + ".activation", "gelu") != "gelu")
            {
                yield return $"The {part} activation '{Text(g, Stt + part + ".activation", "")}' is not supported (GELU only).";
            }
        }

        if (Text(g, Frontend + "type", "mel") != "mel" || Text(g, Frontend + "normalize", "whisper_logmel") != "whisper_logmel")
        {
            yield return $"The audio front end '{Text(g, Frontend + "type", "")}' / '{Text(g, Frontend + "normalize", "")}' is not supported (Whisper's log-mel only).";
        }

        if (g.Tensor("frontend.mel_filterbank") is null)
        {
            yield return "The file stores no mel filter bank ('frontend.mel_filterbank').";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        string a = Architecture + ".";
        int hidden = Int(g, Stt + "decoder.d_model");
        int heads = Int(g, Stt + "decoder.n_heads");
        int headDim = hidden / heads;
        int context = Int(g, Stt + "decoder.max_target_positions", 448);
        int sampleRate = Int(g, Frontend + "sample_rate", 16000);
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "encoder.layer_count"] = Int(g, Stt + "encoder.n_layers"),
            [a + "decoder.layer_count"] = Int(g, Stt + "decoder.n_layers"),
            [a + "attention.head_count"] = heads,
            [a + "attention.head_dim"] = headDim,
            [a + "attention.scale"] = 1f / MathF.Sqrt(headDim),
            [a + "norm"] = "layer",
            [a + "norm_eps"] = 1e-5f,
            [a + "pre_norm"] = true,
            [a + "mlp.activation"] = "gelu",
            [a + "decoder_start_token"] = Int(g, Stt + "decoder_start_token_id"),
            [a + "context_length"] = context,
            [a + "encoder.input"] = "audio",
            [a + "encoder.conv.strides"] = new long[] { 1, 2 },
            [a + "encoder.rows"] = Int(g, Stt + "encoder.max_source_positions", 1500),
            [a + "cross_attention.cached"] = true,
            [a + "audio.sample_rate"] = sampleRate,
            [a + "audio.fft_size"] = Int(g, Frontend + "n_fft", 400),
            [a + "audio.hop_length"] = Int(g, Frontend + "hop_length", 160),
            [a + "audio.mel_bins"] = Int(g, Stt + "encoder.num_mel_bins"),
            [a + "audio.chunk_samples"] = Int(g, Frontend + "n_samples", Int(g, Frontend + "chunk_length", 30) * sampleRate),
        };
    }

    /// <inheritdoc/>
    public string? MapAttachmentName(string ggufName, GgufMetadataView gguf)
    {
        return ggufName switch
        {
            "frontend.mel_filterbank" => MelFilters,
            "frontend.window" => Window,
            _ => null,
        };
    }

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_modelTensors.TryGetValue(name, out var canonical))
        {
            return canonical;
        }

        var match = BlockName().Match(name);
        if (!match.Success)
        {
            return null;
        }

        string stack = match.Groups[1].Value == "enc" ? "encoder" : "decoder";
        return s_blockModules.TryGetValue(match.Groups[3].Value, out var module) ? $"{stack}.layers.{match.Groups[2].Value}.{module}.{match.Groups[4].Value}" : null;
    }

    /// <summary>
    /// The speech-protocol keys from the stored special token ids, and the token types: every id from the end-of-text token on is a
    /// control token (the file types the timestamp tokens as normal text).
    /// </summary>
    public void RefineHyperParameters(Dictionary<string, object> metadata, GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        var tokens = metadata.TryGetValue(CanonicalKeys.TokenizerTokens, out var t) && t is string[] list ? list : [];
        int end = Int(g, "tokenizer.ggml.eos_token_id");
        var types = new int[tokens.Length];
        for (int id = 0; id < tokens.Length; id++)
        {
            types[id] = (int)(id >= end || tokens[id].Length == 0 ? TokenType.Control : TokenType.Normal);
        }

        metadata[CanonicalKeys.TokenizerTokenTypes] = types;
        metadata[CanonicalKeys.TokenizerPre] = "gpt-2";
        metadata[CanonicalKeys.TokenizerAddBos] = false;

        int start = Int(g, Stt + "sot_token_id");
        int noTimestamps = Int(g, Stt + "no_timestamps_token_id");
        metadata[SpeechKeys.StartId] = start;
        metadata[SpeechKeys.EndId] = end;
        metadata[SpeechKeys.NoTimestampsId] = noTimestamps;
        metadata[SpeechKeys.TimestampBeginId] = noTimestamps + 1;
        metadata[SpeechKeys.TimestampStep] = 0.02f;
        metadata[SpeechKeys.MaxTextTokens] = Int(g, Stt + "decoder.max_target_positions", 448) / 2;
        metadata[SpeechKeys.MaxInitialTimestampIndex] = 50;
        if (g.All.ContainsKey(Stt + "prev_sot_token_id"))
        {
            metadata[SpeechKeys.PreviousId] = Int(g, Stt + "prev_sot_token_id");
        }

        int noSpeech = Array.FindIndex(tokens, token => token is "<|nospeech|>" or "<|nocaptions|>");
        if (noSpeech >= 0)
        {
            metadata[SpeechKeys.NoSpeechId] = noSpeech;
        }

        if (tokens.Length >= MultilingualVocabulary)
        {
            int translate = Int(g, Stt + "translate_token_id");
            var languages = Enumerable.Range(start + 1, Math.Max(0, translate - start - 1))
                .Where(id => id < tokens.Length && LanguageToken().IsMatch(tokens[id])).ToArray();
            metadata[SpeechKeys.Languages] = languages.Select(id => tokens[id][2..^2]).ToArray();
            metadata[SpeechKeys.LanguageIds] = languages;
            metadata[SpeechKeys.TranscribeId] = Int(g, Stt + "transcribe_token_id");
            metadata[SpeechKeys.TranslateId] = translate;
        }

        metadata[SpeechKeys.SuppressIds] = IntArray(g, Stt + "suppress_tokens");
        metadata[SpeechKeys.BeginSuppressIds] = IntArray(g, Stt + "begin_suppress_tokens");
    }

    private static int Int(GgufMetadataView g, string key)
        => g.All.TryGetValue(key, out var value) ? Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture) : throw new InvalidDataException($"whisper GGUF: missing '{key}'.");

    private static int Int(GgufMetadataView g, string key, int fallback)
        => g.All.TryGetValue(key, out var value) ? Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture) : fallback;

    private static string Text(GgufMetadataView g, string key, string fallback) => g.All.TryGetValue(key, out var value) && value is string text ? text : fallback;

    private static int[] IntArray(GgufMetadataView g, string key)
    {
        if (!g.All.TryGetValue(key, out var value) || value is not Array array)
        {
            return [];
        }

        var result = new int[array.Length];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = Convert.ToInt32(array.GetValue(i), System.Globalization.CultureInfo.InvariantCulture);
        }

        return result;
    }

    [GeneratedRegex(@"^<\|[a-z]{2,3}\|>$")]
    private static partial Regex LanguageToken();

    [GeneratedRegex(@"^(enc|dec)\.blocks\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex BlockName();

    /// <summary>Per-block modules, transcribe.cpp name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_blockModules = new(StringComparer.Ordinal)
    {
        ["norm_attn"] = "attention_norm",
        ["attn.q"] = "attention.q",
        ["attn.k"] = "attention.k",
        ["attn.v"] = "attention.v",
        ["attn.out"] = "attention.o",
        ["norm_self"] = "attention_norm",
        ["self_attn.q"] = "attention.q",
        ["self_attn.k"] = "attention.k",
        ["self_attn.v"] = "attention.v",
        ["self_attn.out"] = "attention.o",
        ["norm_cross"] = "cross_attention_norm",
        ["cross_attn.q"] = "cross_attention.q",
        ["cross_attn.k"] = "cross_attention.k",
        ["cross_attn.v"] = "cross_attention.v",
        ["cross_attn.out"] = "cross_attention.o",
        ["norm_ffn"] = "mlp_norm",
        ["ffn.fc1"] = "mlp.up",
        ["ffn.fc2"] = "mlp.down",
    };

    /// <summary>Whole-model tensors, transcribe.cpp name to canonical name.</summary>
    private static readonly Dictionary<string, string> s_modelTensors = new(StringComparer.Ordinal)
    {
        ["enc.conv.0.weight"] = "encoder.conv.0.weight",
        ["enc.conv.0.bias"] = "encoder.conv.0.bias",
        ["enc.conv.1.weight"] = "encoder.conv.1.weight",
        ["enc.conv.1.bias"] = "encoder.conv.1.bias",
        ["enc.pos_emb.weight"] = "encoder.position_embedding.weight",
        ["enc.final_norm.weight"] = "encoder.output_norm.weight",
        ["enc.final_norm.bias"] = "encoder.output_norm.bias",
        ["dec.token_embd.weight"] = "token_embedding.weight",
        ["dec.pos_emb.weight"] = "decoder.position_embedding.weight",
        ["dec.final_norm.weight"] = "decoder.output_norm.weight",
        ["dec.final_norm.bias"] = "decoder.output_norm.bias",
        ["dec.output.weight"] = "output.weight",
    };
}
