using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.Library;

/// <summary>Reads a package's capabilities without loading it (graphs, chat template, trained context length, precompiled artifact sets).</summary>
internal static class ModelInspector
{
    /// <summary>
    /// Capabilities, trained context, kind and precompiled artifact sets of a <c>.slm</c> package (the kind its model description names, for packages without
    /// one the resolved default); for other files and unreadable packages no capabilities and the kind their file type suggests
    /// (<see cref="InspectedModel.KindInferred"/>).
    /// </summary>
    public static InspectedModel Inspect(string path)
    {
        if (!path.EndsWith(".slm", StringComparison.OrdinalIgnoreCase) || !File.Exists(path))
        {
            return new InspectedModel(ModelCapabilities.None, null, null, InferKind(path), KindInferred: true, Precompiled: []);
        }

        try
        {
            var package = ModelPackageReader.Open(path);
            var definition = package.Definition;
            var graphs = definition.Graphs.Select(graph => graph.Name).ToHashSet(StringComparer.Ordinal);
            var template = definition.Metadata.TryGetValue(CanonicalKeys.ChatTemplate, out var value) ? value as string : null;
            var capabilities = TemplateCapabilities.From(template);
            if (graphs.Contains(WellKnown.VisionEncodeGraph))
            {
                capabilities |= ModelCapabilities.Vision;
            }

            string kind = KindOf(definition.Description.Kind, graphs, definition.Name, definition.Metadata.TryGetValue(CanonicalKeys.Name, out var general) ? general as string : null);
            if (kind == ModelKindCatalog.Embeddings)
            {
                capabilities = (capabilities & ~ModelCapabilities.Chat) | ModelCapabilities.Embedding;
            }

            int? context = definition.Properties.TryGetValue(WellKnown.ContextLength, out var raw) ? ToInt(raw) : null;
            var precompiled = package.ArtifactSets.Select(set => set.ToString()).ToList();
            return new InspectedModel(capabilities, context, definition.Architecture, kind, KindInferred: false, precompiled);
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            return new InspectedModel(ModelCapabilities.None, null, null, ModelKinds.TextGeneration, KindInferred: true, Precompiled: []);
        }
    }

    /// <summary>
    /// The kind of a package. A declared kind other than text generation (which is also what a package without a description resolves to) wins.
    /// Text generation is refined only where the graphs prove it is not one: no decode graph means an encoder, scoring pairs through the classify
    /// graph (a reranker) or pooling through the embed graph (embeddings). A model with a decode graph stays a language model whatever auxiliary
    /// graphs it carries (the decoder adapters add embed and classify graphs to every model); only a name that says so (Qwen3-Reranker) makes it a reranker.
    /// A declared <c>embeddings</c> kind (older feed packages of cross-encoders) becomes a reranker only for an encoder with a classify graph and no decode graph.
    /// </summary>
    internal static string KindOf(string declared, IReadOnlySet<string> graphs, string name, string? generalName)
    {
        // Packages published before the feed declared cross-encoders as rerankers say "embeddings": an encoder (no decode graph) that scores
        // pairs through a classify graph is a reranker. Embedding models carry the embed graph only.
        if (declared == ModelKindCatalog.Embeddings)
        {
            return graphs.Contains(WellKnown.ClassifyGraph) && !graphs.Contains(WellKnown.DecodeGraph)
                ? ModelKindCatalog.Reranker
                : declared;
        }

        if (declared != ModelKinds.TextGeneration)
        {
            return declared;
        }

        if (!graphs.Contains(WellKnown.DecodeGraph))
        {
            if (graphs.Contains(WellKnown.ClassifyGraph))
            {
                return ModelKindCatalog.Reranker;
            }

            return graphs.Contains(WellKnown.EmbedGraph) ? ModelKindCatalog.Embeddings : declared;
        }

        bool named = name.Contains("rerank", StringComparison.OrdinalIgnoreCase) || generalName?.Contains("rerank", StringComparison.OrdinalIgnoreCase) == true;
        return named && graphs.Contains(WellKnown.ClassifyGraph) ? ModelKindCatalog.Reranker : declared;
    }

    /// <summary>
    /// The kind a model file that is not a package probably has: whisper.cpp model files (by their ggml magic number, whatever their
    /// extension: some repositories ship them as <c>.gguf</c>), GGUF files of the <c>whisper</c> architecture and checkpoint folders of
    /// the speech-recognition families (<see cref="s_speechRecognitionModelTypes"/>) recognize speech, a GGUF named like OuteTTS speaks,
    /// Kokoro and Orpheus folders speak (see <see cref="KindOfConfig"/>), Stable Diffusion checkpoints and the vision checkpoints are recognized by <see cref="ImageModelProbe"/>, other GGUF files and
    /// checkpoint folders are language models.
    /// </summary>
    private static string InferKind(string path)
    {
        if (ImageModelProbe.KindOf(path) is { } imageKind)
        {
            return imageKind;
        }

        if (File.Exists(path) && HasGgmlMagic(path))
        {
            return ModelKinds.SpeechRecognition;
        }

        if (path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            string? architecture = GgufArchitecture(path);
            if (architecture is "whisper")
            {
                return ModelKinds.SpeechRecognition;
            }

            if (architecture is not null && s_encoderGgufArchitectures.Contains(architecture))
            {
                return System.IO.Path.GetFileName(path).Contains("rerank", StringComparison.OrdinalIgnoreCase) || GgufPooling(path, architecture) == 4
                    ? ModelKindCatalog.Reranker
                    : ModelKindCatalog.Embeddings;
            }

            string file = System.IO.Path.GetFileName(path);
            if (file.Contains("outetts", StringComparison.OrdinalIgnoreCase))
            {
                return ModelKinds.SpeechSynthesis;
            }

            // Decoder rerankers (Qwen3-Reranker) are named so; a chat model never becomes one.
            return file.Contains("rerank", StringComparison.OrdinalIgnoreCase) ? ModelKindCatalog.Reranker : ModelKinds.TextGeneration;
        }

        if (IsSileroVad(path))
        {
            return ModelKinds.VoiceActivity;
        }

        string config = System.IO.Path.Combine(path, "config.json");
        if (Directory.Exists(path) && File.Exists(config))
        {
            return KindOfConfig(config, System.IO.Path.GetFileName(path.TrimEnd(System.IO.Path.DirectorySeparatorChar)));
        }

        return ModelKindCatalog.Other;
    }

    /// <summary>Whether <paramref name="path"/> is a Silero VAD ONNX file or a folder holding one.</summary>
    private static bool IsSileroVad(string path)
    {
        try
        {
            if (File.Exists(path))
            {
                string file = System.IO.Path.GetFileName(path);
                return file.StartsWith("silero", StringComparison.OrdinalIgnoreCase) && file.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase);
            }

            return Directory.Exists(path) && Directory.EnumerateFiles(path, "silero*.onnx").Any();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>The kinds of the audio checkpoint folders by their Hugging Face <c>model_type</c>.</summary>
    private static readonly Dictionary<string, string> s_audioModelTypes = new(StringComparer.Ordinal)
    {
        ["qwen3_asr"] = ModelKinds.SpeechRecognition,
        ["encodec"] = ModelKinds.AudioCodec,
        ["mimi"] = ModelKinds.AudioCodec,
        ["dac"] = ModelKinds.AudioCodec,
        ["qwen3_tts_tokenizer_12hz"] = ModelKinds.AudioCodec,
        ["musicgen"] = ModelKinds.MusicGeneration,
        ["clap"] = ModelKinds.AudioEmbedding,
    };

    /// <summary>The Hugging Face <c>model_type</c> values of the speech-recognition checkpoints Sharplight imports.</summary>
    private static readonly HashSet<string> s_speechRecognitionModelTypes = new(StringComparer.Ordinal)
    {
        "whisper", "moonshine", "parakeet_ctc", "parakeet_tdt", "parakeet_rnnt", "canary", "voxtral",
    };

    /// <summary>The <c>general.architecture</c> values of llama.cpp's bidirectional encoders (embedding models and cross-encoder rerankers).</summary>
    private static readonly HashSet<string> s_encoderGgufArchitectures = new(StringComparer.Ordinal)
    {
        "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3",
    };

    /// <summary>The <c>pooling_type</c> of an encoder GGUF file (4 marks a ranking model), or -1.</summary>
    private static long GgufPooling(string path, string architecture)
    {
        try
        {
            using var file = Sharplight.Import.Gguf.GgufFile.Open(path);
            return file.Metadata.TryGetValue($"{architecture}.pooling_type", out var value) ? Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture) : -1;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException or FormatException or InvalidCastException)
        {
            return -1;
        }
    }

    /// <summary>The <c>general.architecture</c> of a GGUF file, or null when it has none or cannot be read.</summary>
    private static string? GgufArchitecture(string path)
    {
        try
        {
            using var file = Sharplight.Import.Gguf.GgufFile.Open(path);
            return file.Architecture;
        }
        catch (Exception error) when (error is IOException or InvalidDataException or NotSupportedException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>Whether the file starts with the ggml magic number of whisper.cpp model files.</summary>
    private static bool HasGgmlMagic(string path)
    {
        try
        {
            using var stream = File.OpenRead(path);
            Span<byte> magic = stackalloc byte[4];
            return stream.Read(magic) == 4 && System.Buffers.Binary.BinaryPrimitives.ReadUInt32LittleEndian(magic) == 0x67676d6c;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            return false;
        }
    }

    /// <summary>
    /// The kind of a checkpoint folder by its <c>config.json</c>: the speech-recognition families recognize speech, a Kokoro configuration (the <c>istftnet</c> section) and a folder
    /// named like Orpheus (a language model that speaks through a codec) synthesize it, everything else is a language model.
    /// </summary>
    private static string KindOfConfig(string config, string folder)
    {
        System.Text.Json.JsonElement root;
        try
        {
            using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(config));
            root = document.RootElement.Clone();
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            return ModelKinds.TextGeneration;
        }

        if (root.ValueKind != System.Text.Json.JsonValueKind.Object)
        {
            return ModelKinds.TextGeneration;
        }

        string? modelType = root.TryGetProperty("model_type", out var type) && type.ValueKind == System.Text.Json.JsonValueKind.String ? type.GetString() : null;
        if (modelType is not null && s_speechRecognitionModelTypes.Contains(modelType))
        {
            return ModelKinds.SpeechRecognition;
        }

        if (modelType is not null && s_audioModelTypes.TryGetValue(modelType, out string? audioKind))
        {
            return audioKind;
        }

        if (folder.Contains("demucs", StringComparison.OrdinalIgnoreCase))
        {
            return ModelKinds.AudioSeparation;
        }

        if (root.TryGetProperty("istftnet", out _) && root.TryGetProperty("plbert", out _) || modelType is "qwen3_tts")
        {
            return ModelKinds.SpeechSynthesis;
        }

        var architectures = root.TryGetProperty("architectures", out var list) && list.ValueKind == System.Text.Json.JsonValueKind.Array
            ? list.EnumerateArray().Select(entry => entry.GetString() ?? "").ToList()
            : [];
        bool encoder = modelType is not null && s_encoderModelTypes.Contains(modelType);
        bool named = folder.Contains("rerank", StringComparison.OrdinalIgnoreCase);
        if (named || encoder && architectures.Any(name => name.EndsWith("ForSequenceClassification", StringComparison.Ordinal)) && LabelCount(root) <= 1)
        {
            // Cross-encoders (one score per pair) and the decoder rerankers (Qwen3-Reranker, scored by a yes/no answer).
            return ModelKindCatalog.Reranker;
        }

        if (architectures.Count > 0 && architectures.All(name => name.EndsWith("Model", StringComparison.Ordinal))
            && (encoder || File.Exists(System.IO.Path.Combine(System.IO.Path.GetDirectoryName(config)!, "modules.json")) || folder.Contains("embed", StringComparison.OrdinalIgnoreCase)))
        {
            return ModelKindCatalog.Embeddings;
        }

        if (folder.Contains("embedding", StringComparison.OrdinalIgnoreCase) && architectures.Any(name => name.EndsWith("ForCausalLM", StringComparison.Ordinal))
            && File.Exists(System.IO.Path.Combine(System.IO.Path.GetDirectoryName(config)!, "modules.json")))
        {
            return ModelKindCatalog.Embeddings;
        }

        return folder.Contains("orpheus", StringComparison.OrdinalIgnoreCase) ? ModelKinds.SpeechSynthesis : ModelKinds.TextGeneration;
    }

    /// <summary>The Hugging Face <c>model_type</c> values of the bidirectional encoders Sharplight imports as embedding models and cross-encoder rerankers.</summary>
    private static readonly HashSet<string> s_encoderModelTypes = new(StringComparer.Ordinal)
    {
        "bert", "xlm-roberta", "roberta", "modernbert", "nomic_bert", "distilbert", "electra", "eurobert", "jina_bert",
    };

    private static int LabelCount(System.Text.Json.JsonElement root)
    {
        if (root.TryGetProperty("num_labels", out var labels) && labels.ValueKind == System.Text.Json.JsonValueKind.Number)
        {
            return labels.GetInt32();
        }

        return root.TryGetProperty("id2label", out var map) && map.ValueKind == System.Text.Json.JsonValueKind.Object ? map.EnumerateObject().Count() : 1;
    }

    private static bool IsSpeechRecognitionConfig(string config)
    {
        try
        {
            using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(config));
            return document.RootElement.TryGetProperty("model_type", out var type) && type.GetString() is { } name && s_speechRecognitionModelTypes.Contains(name);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            return false;
        }
    }

    private static int? ToInt(object value)
    {
        return value switch
        {
            int i => i,
            long l => (int)Math.Min(int.MaxValue, l),
            uint u => (int)Math.Min(int.MaxValue, u),
            ulong ul => (int)Math.Min(int.MaxValue, ul),
            _ => null,
        };
    }
}
