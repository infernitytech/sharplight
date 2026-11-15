using System.Collections;
using System.Globalization;
using Sharplight.Hub;
using Sharplight.Models;
using Sharplight.Package;
using Sharplight.Studio.Models;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>The overview tab: the file, the model (kind and components), where it came from, its tokenizer, chat template and states.</summary>
internal sealed class OverviewViewModel
{
    private const int CommitLength = 12;

    /// <param name="info">The package.</param>
    /// <param name="definition">Its decoded definition, or null when it does not decode.</param>
    /// <param name="definitionError">Why the definition did not decode.</param>
    /// <param name="manifest">The store record when the package was opened from the library.</param>
    /// <param name="kindTotals">Bytes per section kind.</param>
    public OverviewViewModel(PackageInfo info, DefinitionInfo? definition, string? definitionError, ModelManifest? manifest, IReadOnlyList<ShareRow> kindTotals)
    {
        DefinitionError = definitionError;
        KindTotals = kindTotals;
        Package =
        [
            new("File", info.Path),
            new("Format version", info.Version.ToString(CultureInfo.InvariantCulture)),
            new("File size", ByteFormat.Exact(info.FileSize)),
            new("Sections", $"{info.Sections.Count} ({string.Join(", ", info.Sections.GroupBy(s => s.Kind).Select(g => g.Count() == 1 ? g.Key : $"{g.Key} × {g.Count()}"))})"),
            new("Weights", $"{info.Weights.Count:N0} · {ByteFormat.Format(info.WeightBytes)}"),
            new("Artifact sets", info.ArtifactSets.Count == 0 ? "none" : $"{info.ArtifactSets.Count} · {ByteFormat.Format(info.ArtifactSets.Sum(s => s.TotalBytes))}"),
            new("Reclaimable", info.ReclaimableBytes > 0 ? $"{ByteFormat.Exact(info.ReclaimableBytes)} (compacting removes it)" : "none"),
        ];

        var metadata = definition?.Metadata ?? new Dictionary<string, object>();
        var properties = definition?.Properties ?? new Dictionary<string, object>();
        if (definition is null)
        {
            Title = Path.GetFileName(info.Path);
            Model = [new("Definition", "not readable")];
        }
        else
        {
            Title = definition.Name.Length > 0 ? definition.Name : Path.GetFileName(info.Path);
            long parameters = info.Weights.Where(w => w.Shape.IsStatic).Sum(w => w.Shape.Elements());
            Model = Facts(
                ("Name", definition.Name),
                ("Kind", definition.HasDescription ? definition.Description.Kind : $"{definition.Description.Kind} (no description stored)"),
                ("Architecture", definition.Architecture),
                ("Parameters", parameters > 0 ? ByteFormat.Count(parameters) : null),
                ("Graphs", string.Join(", ", definition.Graphs.Select(g => g.Name))),
                ("Nodes", $"{definition.Graphs.Sum(g => g.NodeCount):N0} in {ByteFormat.Count(definition.Graphs.Count, "graph", "graphs")}"),
                ("States", definition.States.Count.ToString(CultureInfo.InvariantCulture)),
                ("Context length", Text(properties, WellKnown.ContextLength)),
                ("Vocabulary", Text(properties, WellKnown.VocabSize)),
                ("Properties", properties.Count.ToString(CultureInfo.InvariantCulture)),
                ("Metadata entries", metadata.Count.ToString(CultureInfo.InvariantCulture)));
        }

        Source = Facts(
            ("Repository", manifest?.Repository),
            ("Variant", manifest is { Tag.Length: > 0 } ? manifest.Tag : null),
            ("Revision", manifest?.Revision),
            ("Commit", manifest?.Commit is { Length: > 0 } commit ? commit[..Math.Min(CommitLength, commit.Length)] : null),
            ("Source format", manifest?.Format.ToString()),
            ("Source files", manifest is { SourceFiles.Count: > 0 } ? $"{manifest.SourceFiles.Count} · {ByteFormat.Format(manifest.SourceBytes)}{(manifest.SourceKept ? " (kept)" : "")}" : null),
            ("Downloaded", manifest is null ? null : manifest.PulledAt.ToLocalTime().ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture)),
            ("Base model", Text(metadata, "general.basename")),
            ("Fine-tune", Text(metadata, "general.finetune")),
            ("Size label", Text(metadata, "general.size_label")),
            ("Model type", Text(metadata, "general.type")),
            ("Source file type", Text(metadata, "general.file_type")),
            ("Quantization version", Text(metadata, "general.quantization_version")),
            ("Graph optimization", Text(properties, "graph.optimized")));

        var tokens = metadata.TryGetValue(CanonicalKeys.TokenizerTokens, out var tokenValue) ? tokenValue as IList : null;
        Tokenizer = Facts(
            ("Model", Text(metadata, CanonicalKeys.TokenizerModel)),
            ("Pre-tokenizer", Text(metadata, CanonicalKeys.TokenizerPre)),
            ("Vocabulary", tokens is null ? null : $"{tokens.Count:N0} tokens"),
            ("Merges", metadata.TryGetValue(CanonicalKeys.TokenizerMerges, out var merges) && merges is IList list ? $"{list.Count:N0}" : null),
            ("Scores", metadata.ContainsKey(CanonicalKeys.TokenizerScores) ? "stored" : null),
            ("Begin of sequence", Token(metadata, CanonicalKeys.TokenizerBosId, tokens)),
            ("End of sequence", Token(metadata, CanonicalKeys.TokenizerEosId, tokens)),
            ("End of turn", Token(metadata, CanonicalKeys.TokenizerEotId, tokens)),
            ("Padding", Token(metadata, CanonicalKeys.TokenizerPadId, tokens)),
            ("Unknown", Token(metadata, CanonicalKeys.TokenizerUnkId, tokens)),
            ("Adds BOS", Text(metadata, CanonicalKeys.TokenizerAddBos)),
            ("Adds EOS", Text(metadata, CanonicalKeys.TokenizerAddEos)),
            ("Adds space prefix", Text(metadata, CanonicalKeys.TokenizerAddSpacePrefix)));
        ChatTemplate = metadata.TryGetValue(CanonicalKeys.ChatTemplate, out var template) ? template as string : null;
        States = definition is null ? [] : [.. definition.States.Select(StateRow.From)];
        var attachmentSizes = info.Attachments.ToDictionary(a => a.Name, a => a.Length, StringComparer.Ordinal);
        Components = definition is null ? [] : [.. definition.Description.Components.Select(c => ComponentRow.From(c, info.Weights, attachmentSizes))];
    }

    /// <summary>The model's components (kind-specific parts: encoder, decoder, tokenizer, denoiser, ...).</summary>
    public IReadOnlyList<ComponentRow> Components { get; }

    /// <summary>True when the model has components to show.</summary>
    public bool HasComponents => Components.Count > 0;

    /// <summary>The model name, or the file name.</summary>
    public string Title { get; }

    /// <summary>Why the definition did not decode, or null.</summary>
    public string? DefinitionError { get; }

    /// <summary>True when the definition did not decode.</summary>
    public bool HasDefinitionError => DefinitionError is not null;

    /// <summary>File facts.</summary>
    public IReadOnlyList<FactRow> Package { get; }

    /// <summary>Bytes per section kind.</summary>
    public IReadOnlyList<ShareRow> KindTotals { get; }

    /// <summary>Model facts.</summary>
    public IReadOnlyList<FactRow> Model { get; }

    /// <summary>Where the model came from and how it was converted.</summary>
    public IReadOnlyList<FactRow> Source { get; }

    /// <summary>True when anything is known about the source.</summary>
    public bool HasSource => Source.Count > 0;

    /// <summary>Tokenizer facts.</summary>
    public IReadOnlyList<FactRow> Tokenizer { get; }

    /// <summary>True when the package carries a tokenizer.</summary>
    public bool HasTokenizer => Tokenizer.Count > 0;

    /// <summary>The chat template, or null.</summary>
    public string? ChatTemplate { get; }

    /// <summary>True when the package carries a chat template.</summary>
    public bool HasChatTemplate => ChatTemplate is not null;

    /// <summary>Length of the chat template.</summary>
    public string ChatTemplateSize => ChatTemplate is null ? "" : $"{ChatTemplate.Length:N0} characters";

    /// <summary>Persistent states.</summary>
    public IReadOnlyList<StateRow> States { get; }

    /// <summary>True when the model has states.</summary>
    public bool HasStates => States.Count > 0;

    private static IReadOnlyList<FactRow> Facts(params (string Label, string? Value)[] facts) =>
        [.. facts.Where(f => !string.IsNullOrEmpty(f.Value)).Select(f => new FactRow(f.Label, f.Value!))];

    private static string? Text(IReadOnlyDictionary<string, object> values, string key) =>
        values.TryGetValue(key, out var value) ? DefinitionInfo.FormatValue(value) : null;

    private static string? Token(IReadOnlyDictionary<string, object> values, string key, IList? tokens)
    {
        if (!values.TryGetValue(key, out var value))
        {
            return null;
        }

        if (value is not (int or long or uint or ulong or short or ushort or byte or sbyte))
        {
            return DefinitionInfo.FormatValue(value);
        }

        long id = Convert.ToInt64(value, CultureInfo.InvariantCulture);
        return tokens is not null && id >= 0 && id < tokens.Count && tokens[(int)id] is string text ? $"{id} {text}" : id.ToString(CultureInfo.InvariantCulture);
    }
}
