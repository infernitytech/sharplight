namespace Sharplight.Studio.ViewModels.Chat;

/// <summary>A picture the chat generated.</summary>
/// <param name="Png">The encoded image.</param>
/// <param name="ModelId">The image model.</param>
/// <param name="Seed">The seed that made it.</param>
/// <param name="Width">Width in pixels.</param>
/// <param name="Height">Height in pixels.</param>
/// <param name="Seconds">How long it took.</param>
internal sealed record GeneratedPicture(byte[] Png, string ModelId, long Seed, int Width, int Height, double Seconds);
