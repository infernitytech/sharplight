using System.Text.Json;
using System.Text.RegularExpressions;
using Sharplight.Models;
using Sharplight.Tensors;

namespace Sharplight.HuggingFace;

/// <summary>
/// Hugging Face Whisper checkpoints (<c>whisper</c>: openai/whisper-*, distil-whisper, whisper-large-v3-turbo) → canonical
/// <c>encoder_decoder</c> with an audio encoder (documented on <c>EncoderDecoderAdapter</c>): the log-mel front end from
/// <c>preprocessor_config.json</c> (filter bank computed as transformers' feature extractor does), the two-convolution stem, pre-norm
/// LayerNorm layers with GELU MLPs, learned decoder positions, the LM head tied to the token embedding, cached cross-attention over the
/// 1500 encoder rows, and the speech-protocol tokens (<see cref="SpeechKeys"/>) from <c>generation_config.json</c> and the vocabulary.
/// </summary>
internal sealed partial class WhisperHfMapping : IHfArchitectureMapping
{
    /// <inheritdoc/>
    public string Architecture => "encoder_decoder";

    /// <inheritdoc/>
    public IReadOnlyList<string> ModelTypes { get; } = ["whisper"];

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(HfConfig c, HfImportOptions options)
    {
        string a = Architecture + ".";
        int hidden = c.Int("d_model");
        int heads = c.Int("encoder_attention_heads");
        if (c.GetInt("decoder_attention_heads") is { } decoderHeads && decoderHeads != heads)
        {
            throw new NotSupportedException("whisper: different encoder and decoder head counts are not supported.");
        }

        string act = c.GetString("activation_function") ?? "gelu";
        if (act != "gelu")
        {
            throw new NotSupportedException($"whisper: activation '{act}' is not supported.");
        }

        using var preprocessor = ReadJson(c.Directory, "preprocessor_config.json");
        var pre = preprocessor?.RootElement;
        int hop = Int(pre, "hop_length", 160);
        int sampleRate = Int(pre, "sampling_rate", 16000);
        int chunkSeconds = Int(pre, "chunk_length", 30);
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
            [a + "audio.sample_rate"] = sampleRate,
            [a + "audio.fft_size"] = Int(pre, "n_fft", 400),
            [a + "audio.hop_length"] = hop,
            [a + "audio.frames"] = Int(pre, "nb_max_frames", chunkSeconds * sampleRate / hop),
        };
    }

    [GeneratedRegex(@"^(encoder|decoder)\.layers\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex LayerName();

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, WeightInfo> MapWeights(HfConfig c, IReadOnlyDictionary<string, SafetensorsTensor> tensors, HfImportOptions options)
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

        int mels = c.Int("num_mel_bins");
        using var preprocessor = ReadJson(c.Directory, "preprocessor_config.json");
        int fft = Int(preprocessor?.RootElement, "n_fft", 400);
        int sampleRate = Int(preprocessor?.RootElement, "sampling_rate", 16000);
        int bins = (fft / 2) + 1;
        collector.Add(WeightInfo.Derived("audio.mel_filters.weight", DataType.F32, new Shape(mels, bins), () =>
        {
            var filters = HfMelFilterBank.Create(mels, fft, sampleRate);
            var bytes = new byte[filters.Length * sizeof(float)];
            Buffer.BlockCopy(filters, 0, bytes, 0, bytes.Length);
            return bytes;
        }));
        return collector.Weights;
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
        if ((GenInt("prev_sot_token_id") ?? Token("<|startofprev|>")) is { } previous)
        {
            metadata[SpeechKeys.PreviousId] = previous;
        }

        if ((Token("<|nospeech|>") ?? Token("<|nocaptions|>")) is { } noSpeech)
        {
            metadata[SpeechKeys.NoSpeechId] = noSpeech;
        }

        if (gen is { } root && root.TryGetProperty("lang_to_id", out var languages) && languages.ValueKind == JsonValueKind.Object)
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

    private static string ConvName(string name) => $"encoder.conv.{(name.Contains("conv1", StringComparison.Ordinal) ? 0 : 1)}.{name[(name.LastIndexOf('.') + 1)..]}";

    private static int[] IntArray(JsonElement? generation, HfConfig config, string key)
    {
        var element = generation is { } g && g.TryGetProperty(key, out var v) ? v : config.Get(key);
        return element is { ValueKind: JsonValueKind.Array } array ? [.. array.EnumerateArray().Select(e => e.GetInt32())] : [];
    }

    private static int Int(JsonElement? root, string key, int fallback)
        => root is { } r && r.TryGetProperty(key, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : fallback;

    private static JsonDocument? ReadJson(string directory, string file)
    {
        string path = Path.Combine(directory, file);
        return File.Exists(path) ? JsonDocument.Parse(File.ReadAllBytes(path)) : null;
    }
}
