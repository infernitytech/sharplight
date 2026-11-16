using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using Sharplight.Audio;
using Sharplight.Models;

namespace Sharplight.Import.Gguf;

/// <summary>
/// GGUF <c>qwen3_asr</c> files as transcribe.cpp writes them (<c>stt.qwen3_asr.encoder.*</c> / <c>decoder.*</c> hyper-parameters, the
/// <c>stt.frontend.*</c> log-mel settings, <c>enc.*</c> audio-tower and <c>dec.*</c> language-model tensors; the repository
/// <c>handy-computer/Qwen3-ASR-1.7B-gguf</c>) → canonical <c>decoder</c> with an audio tower, the model the Hugging Face checkpoint imports
/// to (same weight names, keys and prompt). The stored filter bank and window are not needed: the Slaney bank of the front end's parameters
/// is the <c>mel-filters</c> attachment, as for the Hugging Face import.
/// </summary>
internal sealed partial class Qwen3AsrGgufMapping : IGgufArchitectureMapping
{
    private const string Encoder = "stt.qwen3_asr.encoder.";
    private const string Decoder = "stt.qwen3_asr.decoder.";
    private const string Frontend = "stt.frontend.";
    private const string MelFilters = "feature-extractor/mel-filters";

    /// <inheritdoc/>
    public IReadOnlyList<string> GgufArchitectures { get; } = ["qwen3_asr"];

    /// <inheritdoc/>
    public string Architecture => "decoder";

    /// <inheritdoc/>
    public IEnumerable<string> Validate(GgufMetadataView g)
    {
        if (Text(g, Frontend + "type", "mel") != "mel" || Text(g, Frontend + "normalize", "per_utterance") != "per_utterance")
        {
            yield return "Only the Whisper log-mel front end of Qwen3-ASR is supported.";
        }

        if (Text(g, Decoder + "hidden_act", "silu") != "silu" || Text(g, Encoder + "activation", "gelu") != "gelu")
        {
            yield return "Only the SiLU decoder and the GELU audio tower of Qwen3-ASR are supported.";
        }
    }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, object> MapHyperParameters(GgufMetadataView g)
    {
        const string A = "decoder.";
        int hidden = Int(g, Decoder + "hidden_size");
        int heads = Int(g, Decoder + "n_heads");
        int headDim = Int(g, Decoder + "head_dim", hidden / heads);
        int sampleRate = Int(g, Frontend + "sample_rate", 16000);
        return new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [A + "layer_count"] = Int(g, Decoder + "n_layers"),
            [A + "hidden_size"] = hidden,
            [A + "ffn_hidden_size"] = Int(g, Decoder + "intermediate_size"),
            [A + "norm_eps"] = Float(g, Decoder + "rms_norm_eps", 1e-6f),
            [A + "attention.head_count"] = heads,
            [A + "attention.kv_head_count"] = Int(g, Decoder + "n_kv_heads", heads),
            [A + "attention.head_dim"] = headDim,
            [A + "rope.dim"] = headDim,
            [A + "rope.base"] = Float(g, Decoder + "rope_theta", 1e6f),
            [A + "rope.style"] = "neox",
            [A + "mlp.activation"] = "silu",
            [A + "context_length"] = Int(g, Decoder + "max_position_embeddings", 65536),
            [A + "audio.tower"] = "chunked-conv2d",
            [A + "audio.layer_count"] = Int(g, Encoder + "n_layers"),
            [A + "audio.hidden_size"] = Int(g, Encoder + "d_model"),
            [A + "audio.head_count"] = Int(g, Encoder + "n_heads"),
            [A + "audio.activation"] = "gelu",
            [A + "audio.chunk_frames"] = 2 * Int(g, Encoder + "n_window", 50),
            [A + "audio.window_frames"] = Int(g, Encoder + "n_window_infer", 800),
            [A + "audio.mel_bins"] = Int(g, Frontend + "num_mels", 128),
            [A + "audio.sample_rate"] = sampleRate,
            [A + "audio.fft_size"] = Int(g, Frontend + "n_fft", 400),
            [A + "audio.hop_length"] = Int(g, Frontend + "hop_length", 160),
            [A + "audio.chunk_samples"] = Int(g, Frontend + "n_samples", 30 * sampleRate),
            [A + "audio.min_samples"] = sampleRate / 2,
            [A + "audio.token_id"] = Int(g, "stt.qwen3_asr.audio_token_id"),
            [A + "audio.prompt.template"] = "<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n<|audio_start|>{audio}<|audio_end|><|im_end|>\n<|im_start|>assistant\n",
            [A + "audio.prompt.language_prefix"] = "language {language}<asr_text>",
            [A + "audio.prompt.transcript_marker"] = "<asr_text>",
        };
    }

    /// <summary>
    /// The added tokens after the base vocabulary written as normal tokens (<c>&lt;asr_text&gt;</c>, the transcript marker) become user-defined, as in the
    /// Hugging Face tokenizer, so prompts and outputs treat them as single tokens.
    /// </summary>
    public void RefineHyperParameters(Dictionary<string, object> metadata, GgufMetadataView g, IReadOnlyDictionary<string, WeightInfo> weights)
    {
        if (metadata.GetValueOrDefault(CanonicalKeys.TokenizerTokens) is not string[] tokens || metadata.GetValueOrDefault(CanonicalKeys.TokenizerTokenTypes) is not int[] types)
        {
            return;
        }

        int first = Array.IndexOf(tokens, "<|endoftext|>");
        for (int i = Math.Max(first, 0); i < Math.Min(tokens.Length, types.Length) && first >= 0; i++)
        {
            if (types[i] == 1 && tokens[i].Length > 2 && tokens[i][0] == '<' && tokens[i][^1] == '>' && !tokens[i].StartsWith("<|unused", StringComparison.Ordinal))
            {
                types[i] = 4;
            }
        }
    }

    /// <summary>The stored front-end tensors (filter bank, window) are recomputed by the front end.</summary>
    public bool IgnoresTensor(string ggufName, GgufMetadataView g) => ggufName.StartsWith("frontend.", StringComparison.Ordinal);

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, IWeightSource> SyntheticAttachments(GgufMetadataView g)
    {
        int bins = Int(g, Frontend + "num_mels", 128);
        int fft = Int(g, Frontend + "n_fft", 400);
        int rate = Int(g, Frontend + "sample_rate", 16000);
        return new Dictionary<string, IWeightSource>(StringComparer.Ordinal)
        {
            [MelFilters] = new MemoryWeightSource(() => MemoryMarshal.AsBytes(SlaneyMelFilterBank.Create(bins, fft, rate).AsSpan()).ToArray()),
        };
    }

    /// <inheritdoc/>
    public string? MapWeightName(string name, GgufMetadataView g)
    {
        if (s_globals.TryGetValue(name, out var global))
        {
            return global;
        }

        var conv = Conv().Match(name);
        if (conv.Success)
        {
            return $"audio.conv.{conv.Groups[1].Value}.{conv.Groups[2].Value}";
        }

        var block = Block().Match(name);
        if (!block.Success)
        {
            return null;
        }

        bool audio = block.Groups[1].Value == "enc";
        var modules = audio ? s_audioModules : s_decoderModules;
        return modules.TryGetValue(block.Groups[3].Value, out var module)
            ? $"{(audio ? "audio." : "")}layers.{block.Groups[2].Value}.{module}.{block.Groups[4].Value}"
            : null;
    }

    private static int Int(GgufMetadataView g, string key)
        => g.All.TryGetValue(key, out var value) ? Convert.ToInt32(value, CultureInfo.InvariantCulture) : throw new InvalidDataException($"qwen3_asr GGUF: missing '{key}'.");

    private static int Int(GgufMetadataView g, string key, int fallback)
        => g.All.TryGetValue(key, out var value) ? Convert.ToInt32(value, CultureInfo.InvariantCulture) : fallback;

    private static float Float(GgufMetadataView g, string key, float fallback)
        => g.All.TryGetValue(key, out var value) ? Convert.ToSingle(value, CultureInfo.InvariantCulture) : fallback;

    private static string Text(GgufMetadataView g, string key, string fallback) => g.All.TryGetValue(key, out var value) && value is string text ? text : fallback;

    [GeneratedRegex(@"^enc\.conv\.(\d+)\.(weight|bias)$")]
    private static partial Regex Conv();

    [GeneratedRegex(@"^(enc|dec)\.blocks\.(\d+)\.(.+)\.(weight|bias)$")]
    private static partial Regex Block();

    private static readonly Dictionary<string, string> s_globals = new(StringComparer.Ordinal)
    {
        ["enc.conv_out.weight"] = "audio.conv_out.weight",
        ["enc.ln_post.weight"] = "audio.output_norm.weight",
        ["enc.ln_post.bias"] = "audio.output_norm.bias",
        ["enc.proj1.weight"] = "audio.projector.1.weight",
        ["enc.proj1.bias"] = "audio.projector.1.bias",
        ["enc.proj2.weight"] = "audio.projector.2.weight",
        ["enc.proj2.bias"] = "audio.projector.2.bias",
        ["dec.token_embd.weight"] = "token_embedding.weight",
        ["dec.output_norm.weight"] = "output_norm.weight",
        ["dec.output.weight"] = "lm_head.weight",
    };

    private static readonly Dictionary<string, string> s_audioModules = new(StringComparer.Ordinal)
    {
        ["norm_attn"] = "input_norm",
        ["norm_ffn"] = "post_attention_norm",
        ["attn.q"] = "attention.q",
        ["attn.k"] = "attention.k",
        ["attn.v"] = "attention.v",
        ["attn.out"] = "attention.o",
        ["ffn.fc1"] = "mlp.up",
        ["ffn.fc2"] = "mlp.down",
    };

    private static readonly Dictionary<string, string> s_decoderModules = new(StringComparer.Ordinal)
    {
        ["norm_attn"] = "input_norm",
        ["norm_ffn"] = "post_attention_norm",
        ["attn.q"] = "attention.q",
        ["attn.k"] = "attention.k",
        ["attn.v"] = "attention.v",
        ["attn.o"] = "attention.o",
        ["attn.q_norm"] = "attention.q_norm",
        ["attn.k_norm"] = "attention.k_norm",
        ["ffn.gate"] = "mlp.gate",
        ["ffn.up"] = "mlp.up",
        ["ffn.down"] = "mlp.down",
    };
}
