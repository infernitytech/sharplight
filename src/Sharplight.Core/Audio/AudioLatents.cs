namespace Sharplight.Audio;

/// <summary>The latent representation an audio generator denoised: <see cref="Frames"/> frames of <see cref="Channels"/> values, frame-major.</summary>
/// <param name="Frames">The number of latent frames.</param>
/// <param name="Channels">The values per frame.</param>
/// <param name="Values"><c>Frames · Channels</c> values.</param>
public sealed record AudioLatents(int Frames, int Channels, float[] Values);
