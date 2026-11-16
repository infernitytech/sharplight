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

            string kind = definition.Description.Kind;
            if (graphs.Contains(WellKnown.EmbedGraph) && !graphs.Contains(WellKnown.DecodeGraph))
            {
                capabilities = (capabilities & ~ModelCapabilities.Chat) | ModelCapabilities.Embedding;
                if (kind == ModelKinds.TextGeneration)
                {
                    kind = ModelKindCatalog.Embeddings;
                }
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
    /// The kind a model file that is not a package probably has: whisper.cpp <c>.bin</c> files (by their magic number) and Whisper
    /// checkpoint folders recognize speech, a GGUF named like OuteTTS speaks, Kokoro and Orpheus folders speak (see <see cref="KindOfConfig"/>), Stable Diffusion checkpoints and the vision checkpoints
    /// are recognized by <see cref="ImageModelProbe"/>, other GGUF files and checkpoint folders are language models.
    /// </summary>
    private static string InferKind(string path)
    {
        if (ImageModelProbe.KindOf(path) is { } imageKind)
        {
            return imageKind;
        }

        if (path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            return System.IO.Path.GetFileName(path).Contains("outetts", StringComparison.OrdinalIgnoreCase) ? ModelKinds.SpeechSynthesis : ModelKinds.TextGeneration;
        }

        if (path.EndsWith(".bin", StringComparison.OrdinalIgnoreCase) && HasGgmlMagic(path))
        {
            return ModelKinds.SpeechRecognition;
        }

        string config = System.IO.Path.Combine(path, "config.json");
        if (Directory.Exists(path) && File.Exists(config))
        {
            return KindOfConfig(config, System.IO.Path.GetFileName(path.TrimEnd(System.IO.Path.DirectorySeparatorChar)));
        }

        return ModelKindCatalog.Other;
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
    /// The kind of a checkpoint folder by its <c>config.json</c>: Whisper recognizes speech, a Kokoro configuration (the <c>istftnet</c> section) and a folder named like
    /// Orpheus (a language model that speaks through a codec) synthesize it, everything else is a language model.
    /// </summary>
    private static string KindOfConfig(string config, string folder)
    {
        try
        {
            using var document = System.Text.Json.JsonDocument.Parse(File.ReadAllBytes(config));
            var root = document.RootElement;
            if (root.ValueKind == System.Text.Json.JsonValueKind.Object && root.TryGetProperty("model_type", out var type) && type.GetString() == "whisper")
            {
                return ModelKinds.SpeechRecognition;
            }

            if (root.ValueKind == System.Text.Json.JsonValueKind.Object && root.TryGetProperty("istftnet", out _) && root.TryGetProperty("plbert", out _))
            {
                return ModelKinds.SpeechSynthesis;
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or System.Text.Json.JsonException)
        {
            return ModelKinds.TextGeneration;
        }

        return folder.Contains("orpheus", StringComparison.OrdinalIgnoreCase) ? ModelKinds.SpeechSynthesis : ModelKinds.TextGeneration;
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
