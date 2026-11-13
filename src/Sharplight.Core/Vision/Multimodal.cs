using System.Collections.Concurrent;
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

/// <summary>
/// A prompt after assembly: one row per language-model position-in-cache. Rows are either text tokens or rows of an
/// image's embeddings (<see cref="ImageIndex"/> ≥ 0). Rope positions are absolute and architecture-defined
/// (M-RoPE image spans advance the text position by less than their row count).
/// </summary>
public sealed class MultimodalPrompt
{
    /// <summary>Token id per row (0 for embedding rows).</summary>
    public required int[] Tokens { get; init; }
    /// <summary>Per row: index of the image whose embeddings fill it, or -1 for a text token.</summary>
    public required int[] ImageIndex { get; init; }
    /// <summary>Per row: which embedding row of that image (row-major over the image's output token grid).</summary>
    public required int[] ImageRow { get; init; }
    /// <summary>Rope positions [rows, PositionAxes].</summary>
    public required int[] Positions { get; init; }
    public required int PositionAxes { get; init; }
    /// <summary>Rope position of the next text token after this prompt.</summary>
    public required int NextPosition { get; init; }
    public int Length => Tokens.Length;
}

/// <summary>
/// Architecture-specific prompt assembly for models with media input. Located by canonical architecture id through
/// <see cref="MultimodalRegistry"/>, so it works for models loaded from saved packages (where no adapter runs).
/// </summary>
public interface IMultimodalArchitecture
{
    /// <summary>Canonical architecture ids handled.</summary>
    public IReadOnlyList<string> Architectures { get; }

    /// <summary>
    /// Expands the image placeholders of an already tokenized prompt: <paramref name="tokens"/> holds one placeholder token
    /// per image (in order, as produced by the chat template); each becomes one row per output embedding of that image,
    /// with the positions the architecture defines. The first token gets rope position <paramref name="startPosition"/>.
    /// </summary>
    public MultimodalPrompt Assemble(MultimodalContext context, ReadOnlySpan<int> tokens, IReadOnlyList<PreprocessedImage> images, int startPosition);
}

/// <summary>Registry of <see cref="IMultimodalArchitecture"/>s by canonical architecture id.</summary>
public static class MultimodalRegistry
{
    private static readonly ConcurrentDictionary<string, IMultimodalArchitecture> s_byArch = new(StringComparer.OrdinalIgnoreCase);

    public static void Register(IMultimodalArchitecture m)
    {
        foreach (var a in m.Architectures)
        {
            s_byArch[a] = m;
        }
    }

    public static IMultimodalArchitecture? Find(string architecture) => s_byArch.GetValueOrDefault(architecture);
}

/// <summary>Auxiliary source roles understood by adapters.</summary>
public static class AuxiliaryRoles
{
    /// <summary>A separate vision tower file (llama.cpp's "mmproj").</summary>
    public const string Mmproj = "mmproj";
}
