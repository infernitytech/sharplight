namespace Sharplight.Vision;

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
