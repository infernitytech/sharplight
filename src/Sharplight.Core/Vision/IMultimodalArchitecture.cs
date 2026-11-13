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
    /// <param name="context">The loaded model and its special tokens.</param>
    /// <param name="tokens">Tokenized prompt with one placeholder token per image.</param>
    /// <param name="images">The images in order of their placeholders.</param>
    /// <param name="startPosition">Rope position of the first token.</param>
    /// <exception cref="ArgumentException">The number of placeholders differs from the number of images.</exception>
    public MultimodalPrompt Assemble(MultimodalContext context, ReadOnlySpan<int> tokens, IReadOnlyList<PreprocessedImage> images, int startPosition);
}
