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

    /// <summary>The kind a model file that is not a package probably has: GGUF files and checkpoint folders are language models.</summary>
    private static string InferKind(string path)
    {
        if (path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            return ModelKinds.TextGeneration;
        }

        return Directory.Exists(path) && File.Exists(System.IO.Path.Combine(path, "config.json")) ? ModelKinds.TextGeneration : ModelKindCatalog.Other;
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
