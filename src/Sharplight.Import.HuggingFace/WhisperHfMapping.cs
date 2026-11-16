using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Audio;
using Sharplight.Models;

namespace Sharplight.Import.HuggingFace;

/// <summary>
/// Hugging Face Whisper checkpoints (<c>whisper</c>: openai/whisper-*, distil-whisper, whisper-large-v3-turbo) → canonical
/// <c>encoder_decoder</c> with an audio encoder (documented on <c>EncoderDecoderAdapter</c> and <c>SpeechRecognitionDescription</c>): the
/// log-mel front end from <c>preprocessor_config.json</c> (the filter bank computed as transformers' feature extractor does, as the
/// <c>feature-extractor/mel-filters</c> attachment), the two-convolution stem, pre-norm LayerNorm layers with GELU MLPs, learned decoder
/// positions, the LM head tied to the token embedding, cached cross-attention over the encoder rows, and the speech-protocol tokens
/// (<see cref="SpeechKeys"/>) from <c>generation_config.json</c> and the vocabulary.
/// </summary>
internal sealed partial class WhisperHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["whisper"];

    /// <inheritdoc/>
    public IEnumerable<string> Validate(HfConfig config)
    {
        if (config.GetInt("decoder_attention_heads") is { } decoderHeads && decoderHeads != config.GetInt("encoder_attention_heads"))
        {
            yield return "whisper: different encoder and decoder head counts are not supported.";
        }

        if ((config.GetString("activation_function") ?? "gelu") != "gelu")
        {
            yield return $"whisper: activation '{config.GetString("activation_function")}' is not supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HuggingFaceImportOptions options)
    {
        string a = Architecture + ".";
        int hidden = c.Int("d_model");
        int heads = c.Int("encoder_attention_heads");
        var audio = AudioSettings.Read(c);
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [a + "hidden_size"] = hidden,
            [a + "encoder.layer_count"] = c.Int("encoder_layers"),
            [a + "decoder.layer_count"] = c.Int("decoder_layers"),
            [a + "attention.head_count"] = heads,
            [a + "attention.head_dim"] = hidden / heads,
            [a + "attention.scale"] = 1f / MathF.Sqrt(hidden / heads),
            [a + "norm"] = "layer",
            [a + "norm_eps"] = 1e-5f,
            [a + "pre_norm"] = true,
            [a + "mlp.activation"] = "gelu",
            [a + "decoder_start_token"] = c.Int("decoder_start_token_id"),
            [a + "context_length"] = c.GetInt("max_target_positions") ?? 448,
            [a + "encoder.input"] = "audio",
            [a + "encoder.conv.strides"] = new long[] { 1, 2 },
            [a + "encoder.rows"] = c.GetInt("max_source_positions") ?? 1500,
            [a + "cross_attention.cached"] = true,
            [a + "audio.sample_rate"] = audio.SampleRate,
            [a + "audio.fft_size"] = audio.FftSize,
            [a + "audio.hop_length"] = audio.HopLength,
            [a + "audio.mel_bins"] = c.Int("num_mel_bins"),
            [a + "audio.chunk_samples"] = audio.ChunkSamples,
        };
    }

    [GeneratedRegex(@"^(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex LayerName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HuggingFaceImportOptions options)
    {
        var collector = new HfWeightCollector();
        foreach (var (rawName, tensor) in tensors)
        {
            var info = tensor.ToWeightInfo();
            string name = rawName.StartsWith("model.", StringComparison.Ordinal) ? rawName["model.".Length..] : rawName;
            switch (name)
            {
                case "decoder.embed_tokens.weight":
                    collector.Add(info.As("token_embedding.weight"));
                    continue;
                case "proj_out.weight":
                    collector.Add(info.As("output.weight"));
                    continue;
                case "encoder.embed_positions.weight" or "decoder.embed_positions.weight":
                    collector.Add(HfTransforms.ToF32(info, name[..name.IndexOf('.', StringComparison.Ordinal)] + ".position_embedding.weight"));
                    continue;
                case "encoder.layer_norm.weight" or "encoder.layer_norm.bias" or "decoder.layer_norm.weight" or "decoder.layer_norm.bias":
                    collector.Add(HfTransforms.Plain(info, name.Replace("layer_norm", "output_norm", StringComparison.Ordinal)));
                    continue;
                case "encoder.conv1.weight" or "encoder.conv2.weight":
                    collector.Add(info.As(ConvName(name)));
                    continue;
                case "encoder.conv1.bias" or "encoder.conv2.bias":
                    collector.Add(HfTransforms.Plain(info, ConvName(name)));
                    continue;
            }

            var m = LayerName().Match(name);
            if (!m.Success || !BartHfMapping.LayerModules.TryGetValue(m.Groups[3].Value, out var module))
            {
                throw new NotSupportedException($"whisper: tensor '{rawName}' has no canonical mapping.");
            }

            string canonical = $"{m.Groups[1].Value}.layers.{m.Groups[2].Value}.{module}.{m.Groups[4].Value}";
            collector.Add(info.Shape.Rank == 1 ? HfTransforms.Plain(info, canonical) : info.As(canonical));
        }

        return collector.Weights;
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, IWeightSource> MapAttachments(HfConfig config, string directory)
    {
        int mels = config.Int("num_mel_bins");
        var audio = AudioSettings.Read(config);
        return new Dictionary<string, IWeightSource>(StringComparer.Ordinal)
        {
            ["feature-extractor/mel-filters"] = new MemoryWeightSource(
                () => MemoryMarshal.AsBytes(SlaneyMelFilterBank.Create(mels, audio.FftSize, audio.SampleRate).AsSpan()).ToArray()),
        };
    }

    /// <inheritdoc/>
    public void AddMetadata(IDictionary<string, object> metadata, HfConfig config, string directory)
    {
        var tokens = metadata.TryGetValue(CanonicalKeys.TokenizerTokens, out var t) && t is string[] list ? list : [];
        var ids = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < tokens.Length; i++)
        {
            ids.TryAdd(tokens[i], i);
        }

        using var generation = ReadJson(directory, "generation_config.json");
        var gen = generation?.RootElement;
        int? Token(string text) => ids.TryGetValue(text, out int id) ? id : null;
        int? GenInt(string key) => gen is { } g && g.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;

        metadata[SpeechKeys.StartId] = config.Int("decoder_start_token_id");
        metadata[SpeechKeys.EndId] = GenInt("eos_token_id") ?? config.Int("eos_token_id");
        int noTimestamps = GenInt("no_timestamps_token_id") ?? Token("<|notimestamps|>") ?? throw new NotSupportedException("whisper: no <|notimestamps|> token.");
        metadata[SpeechKeys.NoTimestampsId] = noTimestamps;
        metadata[SpeechKeys.TimestampBeginId] = noTimestamps + 1;
        metadata[SpeechKeys.TimestampStep] = 0.02f;
        metadata[SpeechKeys.MaxTextTokens] = (config.GetInt("max_target_positions") ?? 448) / 2;
        if (GenInt("max_initial_timestamp_index") is { } maxInitial)
        {
            metadata[SpeechKeys.MaxInitialTimestampIndex] = maxInitial;
        }

        if ((GenInt("prev_sot_token_id") ?? Token("<|startofprev|>")) is { } previous)
        {
            metadata[SpeechKeys.PreviousId] = previous;
        }

        if ((Token("<|nospeech|>") ?? Token("<|nocaptions|>")) is { } noSpeech)
        {
            metadata[SpeechKeys.NoSpeechId] = noSpeech;
        }

        if (gen is { } root && root.TryGetProperty("lang_to_id", out var languages) && languages.ValueKind == JsonValueKind.Object
            && GenBool(root, "is_multilingual") != false)
        {
            var ordered = languages.EnumerateObject().Select(p => (Code: p.Name.Trim('<', '>', '|'), Id: p.Value.GetInt32())).OrderBy(p => p.Id).ToArray();
            metadata[SpeechKeys.Languages] = ordered.Select(p => p.Code).ToArray();
            metadata[SpeechKeys.LanguageIds] = ordered.Select(p => p.Id).ToArray();
            if (root.TryGetProperty("task_to_id", out var tasks))
            {
                metadata[SpeechKeys.TranscribeId] = tasks.GetProperty("transcribe").GetInt32();
                metadata[SpeechKeys.TranslateId] = tasks.GetProperty("translate").GetInt32();
            }
        }

        metadata[SpeechKeys.SuppressIds] = IntArray(gen, config, "suppress_tokens");
        metadata[SpeechKeys.BeginSuppressIds] = IntArray(gen, config, "begin_suppress_tokens");
    }

    private static bool? GenBool(JsonElement root, string key)
        => root.TryGetProperty(key, out var v) && v.ValueKind is JsonValueKind.True or JsonValueKind.False ? v.GetBoolean() : null;

    private static string ConvName(string name) => $"encoder.conv.{(name.Contains("conv1", StringComparison.Ordinal) ? 0 : 1)}.{name[(name.LastIndexOf('.') + 1)..]}";

    private static int[] IntArray(JsonElement? generation, HfConfig config, string key)
    {
        var element = generation is { } g && g.TryGetProperty(key, out var v) ? v : config.Get(key);
        return element is { ValueKind: JsonValueKind.Array } array ? [.. array.EnumerateArray().Select(e => e.GetInt32())] : [];
    }

    private static JsonDocument? ReadJson(string directory, string file)
    {
        string path = Path.Combine(directory, file);
        return File.Exists(path) ? JsonDocument.Parse(File.ReadAllBytes(path)) : null;
    }

    /// <summary>The front-end settings of <c>preprocessor_config.json</c> (transformers' defaults when absent).</summary>
    private readonly record struct AudioSettings(int SampleRate, int FftSize, int HopLength, int ChunkSamples)
    {
        public static AudioSettings Read(HfConfig config)
        {
            using var preprocessor = ReadJson(config.Directory, "preprocessor_config.json");
            var root = preprocessor?.RootElement;
            int Int(string key, int fallback) => root is { } r && r.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;
            int sampleRate = Int("sampling_rate", 16000);
            return new AudioSettings(sampleRate, Int("n_fft", 400), Int("hop_length", 160), Int("n_samples", Int("chunk_length", 30) * sampleRate));
        }
    }
}
