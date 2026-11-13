using Sharplight.Models;

namespace Sharplight.Vision;

/// <summary>
/// What an <see cref="IMultimodalArchitecture"/> needs to know about the loaded model while assembling a prompt:
/// the built definition (properties, vision parameters) and a way to find special tokens by their text.
/// </summary>
public sealed record MultimodalContext(ModelDefinition Model, Func<string, int?> LookupToken)
{
    /// <summary>The id of a special token, or an error naming what the model lacks.</summary>
    public int RequireToken(string text) => LookupToken(text) ?? throw new InvalidOperationException($"The tokenizer has no token '{text}'.");
}
