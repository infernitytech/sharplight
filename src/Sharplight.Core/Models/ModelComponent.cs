using System.Text.Json;

namespace Sharplight.Models;

/// <summary>
/// One part of a model (<see cref="ModelDescription"/>): e.g. the encoder, decoder, tokenizer and feature extractor of a speech
/// recognizer, or the text encoder, denoiser, VAE and scheduler of an image generator. A component names the graphs it runs, the
/// weights it owns, its configuration and its attachments; all of them live in the one <see cref="ModelDefinition"/> (one
/// <c>.slm</c> package), so every component keeps the memory-mapped, precompilable fast path. Nothing here is specific to a model
/// kind: roles and configuration keys are open strings.
/// </summary>
public sealed class ModelComponent
{
    /// <summary>Component role (<see cref="ModelComponentRoles"/>; open-ended, lowercase and hyphenated); unique within a model.</summary>
    public required string Role { get; init; }

    /// <summary>
    /// Names of the component's graphs in <see cref="ModelDefinition.Graphs"/> (e.g. <c>encode</c>, or <c>vae_decode</c> and
    /// <c>vae_encode</c>); empty for configuration-only components such as a tokenizer or scheduler. Precompilation compiles each.
    /// </summary>
    public IReadOnlyList<string> Graphs { get; init; } = [];

    /// <summary>
    /// Name prefixes of the weights the component owns (e.g. <c>vae.</c>), for per-component accounting and placement; empty when the
    /// component owns no weights of its own or the model does not partition them.
    /// </summary>
    public IReadOnlyList<string> WeightPrefixes { get; init; } = [];

    /// <summary>
    /// Prefix of the component's canonical metadata keys in <see cref="ModelDefinition.SourceMetadata"/> (e.g. <c>text-encoder-2.</c>
    /// before <c>tokenizer.tokens</c>), or <see langword="null"/> when it uses the model's unprefixed metadata.
    /// </summary>
    public string? MetadataPrefix { get; init; }

    /// <summary>
    /// Component configuration as JSON values (e.g. a scheduler's <c>predictionType</c> and <c>betaSchedule</c>, a feature extractor's
    /// <c>sampleRate</c> and <c>melBins</c>). The keys belong to the importer and task implementation that use them.
    /// </summary>
    public IReadOnlyDictionary<string, JsonElement> Config { get; init; } = new Dictionary<string, JsonElement>();

    /// <summary>
    /// The component's attachments: a local name (e.g. <c>vocab</c>, <c>mel-filters</c>, <c>voices</c>) mapped to the name of an entry
    /// of <see cref="ModelDefinition.Attachments"/> (e.g. <c>tokenizer/vocab.json</c>).
    /// </summary>
    public IReadOnlyDictionary<string, string> Attachments { get; init; } = new Dictionary<string, string>();

    /// <summary>
    /// Default upper bounds of the dynamic symbols of the component's graphs (e.g. the frame count of an audio window, the latent
    /// height and width), used when the component is precompiled or compiled without explicit bounds.
    /// </summary>
    public IReadOnlyDictionary<string, long> SymbolBounds { get; init; } = new Dictionary<string, long>();

    /// <summary>True when <paramref name="weightName"/> starts with one of <see cref="WeightPrefixes"/>.</summary>
    /// <param name="weightName">A weight name of the model.</param>
    public bool OwnsWeight(string weightName) => WeightPrefixes.Any(prefix => weightName.StartsWith(prefix, StringComparison.Ordinal));

    /// <summary>
    /// The component's canonical metadata: the entries of <paramref name="sourceMetadata"/> under <see cref="MetadataPrefix"/> with the
    /// prefix removed, or <paramref name="sourceMetadata"/> itself when the component has no prefix.
    /// </summary>
    /// <param name="sourceMetadata">The model's <see cref="ModelDefinition.SourceMetadata"/>.</param>
    public IReadOnlyDictionary<string, object> Metadata(IReadOnlyDictionary<string, object> sourceMetadata)
    {
        if (MetadataPrefix is null)
        {
            return sourceMetadata;
        }

        var scoped = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in sourceMetadata)
        {
            if (key.StartsWith(MetadataPrefix, StringComparison.Ordinal))
            {
                scoped[key[MetadataPrefix.Length..]] = value;
            }
        }

        return scoped;
    }

    /// <summary>Reads a configuration value, or returns <paramref name="defaultValue"/> when it is absent.</summary>
    /// <typeparam name="T">A type the JSON value deserializes to without reflection: string, bool, a number type.</typeparam>
    /// <param name="key">The configuration key.</param>
    /// <param name="defaultValue">The value for an absent key.</param>
    /// <exception cref="FormatException">The value has another JSON type.</exception>
    public T ConfigValue<T>(string key, T defaultValue) => Config.TryGetValue(key, out var value) ? JsonValues.Convert<T>(value) : defaultValue;
}
