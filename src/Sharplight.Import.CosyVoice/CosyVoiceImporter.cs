using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Sharplight.Audio;
using Sharplight.Import.HuggingFace;
using Sharplight.Import.Onnx;
using Sharplight.Models;

namespace Sharplight.Import.CosyVoice;

/// <summary>
/// CosyVoice importer: maps the Fun-CosyVoice3-0.5B release folder (<c>cosyvoice3.yaml</c>, <c>llm.pt</c>, <c>flow.pt</c>, <c>hift.pt</c>, the Qwen
/// tokenizer in <c>CosyVoice-BlankEN/</c>, and for voice cloning <c>speech_tokenizer_v3.onnx</c> and <c>campplus.onnx</c>) onto the canonical
/// <see cref="ModelSource"/> of architecture <c>cosyvoice3</c>.
/// </summary>
/// <remarks>
/// <para>The speech-token language model's Qwen2 backbone is mapped like a Hugging Face Qwen2 checkpoint to the canonical decoder names: its speech-token
/// embedding (6561 tokens plus 200 control rows) is the <c>token_embedding</c> and the speech head (<c>llm_decoder</c>) the <c>lm_head</c>; the text
/// embedding stays on the host as <c>cosyvoice3.text_embedding</c>. The flow (token embedding, pre-lookahead convolutions, speaker projection, the DiT
/// estimator) keeps its names under <c>flow.</c>, the causal HiFT vocoder under <c>hift.</c> with the weight norms folded. The ONNX S3 tokenizer v3 and
/// CAM++ become weights under <c>speech_tokenizer.</c> and <c>campplus.</c> named by module path (<see cref="CosyVoiceOnnxMapping"/>), so the generic
/// tokenizer and speaker-encoder graphs run them and a saved package carries them. The Qwen tokenizer gets CosyVoice 3's added special tokens
/// (<c>&lt;|endofprompt|&gt;</c>, the breath and noise marks, the phoneme tokens) in the release's order.</para>
/// <para>Register with <c>services.AddCosyVoiceImporter()</c> so that the <see cref="ModelLoader"/> opens CosyVoice folders, or call <see cref="Open"/>.</para>
/// </remarks>
public sealed partial class CosyVoiceImporter : IModelImporter
{
    /// <summary>The canonical architecture id.</summary>
    public const string Architecture = "cosyvoice3";

    private const string Gain = ".parametrizations.weight.original0";
    private const string Direction = ".parametrizations.weight.original1";
    private const string TokenizerFolder = "CosyVoice-BlankEN";

    /// <inheritdoc/>
    public string Format => "cosyvoice";

    /// <inheritdoc/>
    public int Priority => 100;

    /// <inheritdoc/>
    public bool CanOpen(string path)
        => Directory.Exists(path) && File.Exists(Path.Combine(path, "cosyvoice3.yaml")) && File.Exists(Path.Combine(path, "llm.pt"))
           && File.Exists(Path.Combine(path, "flow.pt")) && File.Exists(Path.Combine(path, "hift.pt")) && Directory.Exists(Path.Combine(path, TokenizerFolder));

    /// <inheritdoc/>
    ModelSource IModelImporter.Open(string path) => Open(path);

    /// <summary>Opens a Fun-CosyVoice3 release folder.</summary>
    /// <param name="path">The folder.</param>
    /// <exception cref="InvalidDataException">The folder is not a CosyVoice 3 release.</exception>
    public static ModelSource Open(string path)
    {
        var owned = new List<IDisposable>();
        try
        {
            var weights = new Dictionary<string, WeightInfo>(StringComparer.Ordinal);
            var metadata = new Dictionary<string, object>(StringComparer.Ordinal);
            string tokenizerDir = Path.Combine(path, TokenizerFolder);
            var llm = SafetensorsFile.OpenTorch(Path.Combine(path, "llm.pt"), new Sharplight.Import.PyTorch.TorchCheckpointOptions());
            owned.Add(llm);
            int speechRows = MapLanguageModel(llm, tokenizerDir, weights, metadata);

            var flow = SafetensorsFile.OpenTorch(Path.Combine(path, "flow.pt"), new Sharplight.Import.PyTorch.TorchCheckpointOptions());
            owned.Add(flow);
            MapFolded(flow, "flow.", weights);
            var hift = SafetensorsFile.OpenTorch(Path.Combine(path, "hift.pt"), new Sharplight.Import.PyTorch.TorchCheckpointOptions());
            owned.Add(hift);
            MapFolded(hift, "hift.", weights);

            using (var tokenizerConfig = HfConfig.Load(tokenizerDir))
            {
                AddTokenizer(metadata, tokenizerDir, tokenizerConfig);
            }

            var attachments = new Dictionary<string, IWeightSource>(StringComparer.Ordinal);
            string tokenizerOnnx = Path.Combine(path, "speech_tokenizer_v3.onnx");
            string campplusOnnx = Path.Combine(path, "campplus.onnx");
            if (File.Exists(tokenizerOnnx) && File.Exists(campplusOnnx))
            {
                MapOnnx(tokenizerOnnx, "speech_tokenizer.", SpeechTokenizerName, weights, owned);
                MapOnnx(campplusOnnx, "campplus.", static n => n, weights, owned);
                AddCloningFilters(attachments);
            }

            metadata[CanonicalKeys.Architecture] = Architecture;
            metadata[CanonicalKeys.Name] = Path.GetFileName(path.TrimEnd(Path.DirectorySeparatorChar));
            AddHyperParameters(metadata, File.ReadAllText(Path.Combine(path, "cosyvoice3.yaml")), speechRows);
            return new ModelSource
            {
                Format = "cosyvoice",
                Architecture = Architecture,
                Metadata = metadata,
                Weights = weights,
                Attachments = attachments,
                OwnedResources = owned,
                Path = path,
            };
        }
        catch
        {
            foreach (var resource in owned)
            {
                resource.Dispose();
            }

            throw;
        }
    }

    // The Qwen2 backbone through the Hugging Face decoder mapping, with the speech embedding and head as its vocabulary; returns the speech rows.
    private static int MapLanguageModel(SafetensorsFile file, string tokenizerDir, Dictionary<string, WeightInfo> weights, Dictionary<string, object> metadata)
    {
        const string Backbone = "llm.model.model.";
        var qwen = new Dictionary<string, SafetensorsTensor>(StringComparer.Ordinal);
        int speechRows = 0;
        foreach (var (name, tensor) in file.Tensors)
        {
            if (name == Backbone + "embed_tokens.weight")
            {
                weights[Architecture + ".text_embedding"] = tensor.ToWeightInfo().As(Architecture + ".text_embedding");
            }
            else if (name.StartsWith(Backbone, StringComparison.Ordinal))
            {
                qwen["model." + name[Backbone.Length..]] = tensor;
            }
            else if (name == "speech_embedding.weight")
            {
                qwen["model.embed_tokens.weight"] = tensor;
                speechRows = (int)tensor.Shape[0];
            }
            else if (name == "llm_decoder.weight")
            {
                qwen["lm_head.weight"] = tensor;
            }
        }

        if (speechRows == 0 || !qwen.ContainsKey("lm_head.weight"))
        {
            throw new InvalidDataException("llm.pt: not a CosyVoice 3 language model (no speech_embedding / llm_decoder).");
        }

        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(tokenizerDir, "config.json")))!.AsObject();
        config["vocab_size"] = speechRows;
        config["tie_word_embeddings"] = false;
        using var hfConfig = HfConfig.Parse(config.ToJsonString(), tokenizerDir);
        var mapping = new DecoderHfMapping();
        var options = new HuggingFaceImportOptions();
        foreach (var (key, value) in mapping.MapHyperParameters(hfConfig, options))
        {
            metadata[key] = value;
        }

        foreach (var (name, w) in mapping.MapWeights(hfConfig, qwen, options))
        {
            weights[name] = w;
        }

        return speechRows;
    }

    // Every tensor under the prefix, weight-normalized ones folded into a plain weight.
    private static void MapFolded(SafetensorsFile file, string prefix, Dictionary<string, WeightInfo> weights)
    {
        foreach (var (name, tensor) in file.Tensors)
        {
            if (name.EndsWith(Direction, StringComparison.Ordinal) || tensor.Type is null)
            {
                continue;
            }

            if (name.EndsWith(Gain, StringComparison.Ordinal))
            {
                string module = name[..^Gain.Length];
                string canonical = prefix + module + ".weight";
                weights[canonical] = HfWeightNormFolding.Fold(tensor.ToWeightInfo(), file.Tensors[module + Direction].ToWeightInfo(), canonical);
                continue;
            }

            weights[prefix + name] = tensor.ToWeightInfo().As(prefix + name);
        }
    }

    private static void MapOnnx(string file, string prefix, Func<string, string> rename, Dictionary<string, WeightInfo> weights, List<IDisposable> owned)
    {
        var source = OnnxImporter.Open(file);
        owned.AddRange(source.OwnedResources);
        foreach (var (name, weight) in CosyVoiceOnnxMapping.Map(source))
        {
            string canonical = prefix + rename(name);
            weights[canonical] = weight.As(canonical);
        }
    }

    // The S3 tokenizer v3's module names in the layout of the S3 tokenizer graph (s3tokenizer's S3TokenizerV3: encoder.*, quantizer._codebook.*).
    private static string SpeechTokenizerName(string name)
        => name.StartsWith("quantizer.project_in.", StringComparison.Ordinal)
            ? "quantizer._codebook.project_down." + name["quantizer.project_in.".Length..]
            : "encoder." + name;

    // CosyVoice3Tokenizer: the Qwen tokenizer plus the release's special tokens, appended in order (ids from the tokenizer's length on, the ones it
    // already has skipped), as transformers' add_special_tokens assigns them.
    private static void AddTokenizer(Dictionary<string, object> metadata, string dir, HfConfig config)
    {
        var tokenizer = JsonNode.Parse(HfTokenizerTranslator.ConvertSlowByteLevelBpe(dir))!.AsObject();
        var added = tokenizer["added_tokens"]!.AsArray();
        var known = new HashSet<string>(added.Select(t => t!["content"]!.GetValue<string>()), StringComparer.Ordinal);
        int vocabulary = tokenizer["model"]!["vocab"]!.AsObject().Count;
        int next = Math.Max(vocabulary, added.Select(t => t!["id"]!.GetValue<int>() + 1).DefaultIfEmpty(0).Max());
        foreach (string token in CosyVoiceSpecialTokens.All)
        {
            if (!known.Add(token))
            {
                continue;
            }

            added.Add(new JsonObject
            {
                ["id"] = next++,
                ["content"] = token,
                ["single_word"] = false,
                ["lstrip"] = false,
                ["rstrip"] = false,
                ["normalized"] = false,
                ["special"] = true,
            });
        }

        HfTokenizerTranslator.AddFromJson(metadata, System.Text.Encoding.UTF8.GetBytes(tokenizer.ToJsonString()), dir, config);
    }

    // The cloning front ends' filter banks: the S3 tokenizer's Whisper log-mel (128 Slaney bins at 16 kHz, n_fft 400) and the flow prompt's Matcha
    // log-mel (80 Slaney bins up to the Nyquist frequency at 24 kHz, n_fft 1920; the release's fmax is null).
    private static void AddCloningFilters(Dictionary<string, IWeightSource> attachments)
    {
        static MemoryWeightSource Bank(int bins, int fftSize, int sampleRate, double? maxFrequency)
            => new(() => MemoryMarshal.AsBytes(SlaneyMelFilterBank.Create(bins, fftSize, sampleRate, 0.0, maxFrequency).AsSpan()).ToArray());
        attachments["speech-token-features/mel-filters"] = Bank(128, 400, 16000, null);
        attachments["prompt-features/mel-filters"] = Bank(80, 1920, 24000, null);
    }

    // The release's settings: sampling and flow settings read from cosyvoice3.yaml where it states them, the fixed structure of the release otherwise.
    private static void AddHyperParameters(Dictionary<string, object> md, string yaml, int speechRows)
    {
        const string A = Architecture + ".";
        double Number(string key, double fallback)
        {
            var match = Regex.Match(yaml, $@"^\s*{Regex.Escape(key)}:\s*([-+0-9.eE]+)\s*(#.*)?$", RegexOptions.Multiline);
            return match.Success ? double.Parse(match.Groups[1].Value, CultureInfo.InvariantCulture) : fallback;
        }

        int speechTokens = (int)Number("speech_token_size", 6561);
        md[A + "sample_rate"] = (int)Number("sample_rate", 24000);
        md[A + "llm.speech_tokens"] = speechTokens;
        md[A + "llm.speech_rows"] = speechRows;
        md[A + "llm.sos"] = speechTokens;
        md[A + "llm.eos"] = speechTokens + 1;
        md[A + "llm.task_id"] = speechTokens + 2;
        md[A + "llm.top_p"] = Number("top_p", 0.8);
        md[A + "llm.top_k"] = (int)Number("top_k", 25);
        md[A + "llm.window"] = (int)Number("win_size", 10);
        md[A + "llm.tau_r"] = Number("tau_r", 0.1);
        md[A + "llm.min_token_text_ratio"] = 2.0;
        md[A + "llm.max_token_text_ratio"] = 20.0;
        md[A + "flow.token_mel_ratio"] = (int)Number("token_mel_ratio", 2);
        md[A + "flow.pre_lookahead"] = (int)Number("pre_lookahead_len", 3);
        md[A + "flow.depth"] = (int)Number("depth", 22);
        md[A + "flow.heads"] = (int)Number("heads", 16);
        md[A + "flow.mel_bins"] = 80;
        md[A + "flow.steps"] = 10;
        md[A + "flow.guidance"] = Number("inference_cfg_rate", 0.7);
        md[A + "flow.chunk_frames"] = (int)Number("chunk_size", 25) * (int)Number("token_mel_ratio", 2);
        md[A + "hift.upsample_rates"] = new long[] { 8, 5, 3 };
        md[A + "hift.upsample_kernels"] = new long[] { 16, 11, 7 };
        md[A + "hift.fft_size"] = 16;
        md[A + "hift.hop"] = 4;
        md[A + "hift.harmonics"] = (int)Number("nb_harmonics", 8);
        md[A + "hift.look_right"] = (int)Number("conv_pre_look_right", 4);
    }
}
