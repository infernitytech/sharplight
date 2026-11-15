using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.Services.Library;

/// <summary>Reads a package's capabilities without loading it (graphs, chat template, trained context length, precompiled artifact sets).</summary>
internal static class ModelInspector
{
    /// <summary>Capabilities, trained context and precompiled sets of a <c>.slm</c> package; nothing for other files or unreadable packages.</summary>
    public static PackageFacts Inspect(string path)
    {
        if (!path.EndsWith(".slm", StringComparison.OrdinalIgnoreCase) || !File.Exists(path))
        {
            return PackageFacts.None;
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

            if (graphs.Contains(WellKnown.EmbedGraph) && !graphs.Contains(WellKnown.DecodeGraph))
            {
                capabilities = (capabilities & ~ModelCapabilities.Chat) | ModelCapabilities.Embedding;
            }

            int? context = definition.Properties.TryGetValue(WellKnown.ContextLength, out var raw) ? ToInt(raw) : null;
            var precompiled = package.ArtifactSets.Select(set => set.ToString()).ToList();
            return new PackageFacts(capabilities, context, definition.Architecture, precompiled);
        }
        catch (Exception error) when (error is IOException or InvalidDataException or UnauthorizedAccessException)
        {
            return PackageFacts.None;
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
