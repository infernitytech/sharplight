namespace Sharplight.Tasks;

/// <summary>
/// One component of a model that a task is built from: e.g. the text encoder, denoiser and VAE decoder of an image generator, or the
/// audio encoder and text decoder of a speech recognizer. All components live in one <see cref="Models.ModelDefinition"/> (one
/// package): each names the graph it runs and, when it has its own tokenizer or other canonical metadata, the prefix under which that
/// metadata is stored in <see cref="Models.ModelDefinition.SourceMetadata"/>.
/// </summary>
/// <param name="Role">Component role (<see cref="TaskComponentRoles"/>; open-ended, lowercase with underscores).</param>
/// <param name="Graph">Name of the component's graph in <see cref="Models.ModelDefinition.Graphs"/>, or <see langword="null"/> for a
/// metadata-only component (e.g. a second tokenizer).</param>
/// <param name="MetadataPrefix">Prefix of the component's canonical metadata keys (e.g. <c>text_encoder_2.</c> followed by
/// <c>tokenizer.tokens</c>), or <see langword="null"/> when the component uses the model's unprefixed metadata.</param>
public sealed record TaskComponent(string Role, string? Graph = null, string? MetadataPrefix = null)
{
    /// <summary>
    /// The component's canonical metadata: the entries of <paramref name="sourceMetadata"/> under <see cref="MetadataPrefix"/> with the
    /// prefix removed, or <paramref name="sourceMetadata"/> itself when the component has no prefix. Feed it to e.g.
    /// <c>TokenizerData.FromMetadata</c> to build the component's tokenizer.
    /// </summary>
    /// <param name="sourceMetadata">The model's <see cref="Models.ModelDefinition.SourceMetadata"/>.</param>
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
}
