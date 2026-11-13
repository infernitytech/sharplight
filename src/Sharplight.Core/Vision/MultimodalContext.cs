using Sharplight.Models;

namespace Sharplight.Vision;

/// <summary>
/// What an <see cref="IMultimodalArchitecture"/> needs to know about the loaded model while assembling a prompt:
/// the built definition (properties, vision parameters) and a way to find special tokens by their text.
/// </summary>
/// <param name="Model">The built model definition (properties, vision parameters).</param>
/// <param name="LookupToken">Finds a special token id by its text; returns <see langword="null"/> when the tokenizer has none.</param>
public sealed record MultimodalContext(ModelDefinition Model, Func<string, int?> LookupToken)
{
    /// <summary>The id of a special token, or an error naming what the model lacks.</summary>
    /// <param name="text">The token text.</param>
    /// <exception cref="InvalidOperationException">The tokenizer has no such token.</exception>
    public int RequireToken(string text) => LookupToken(text) ?? throw new InvalidOperationException($"The tokenizer has no token '{text}'.");
}
