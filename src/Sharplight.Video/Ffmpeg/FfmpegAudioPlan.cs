namespace Sharplight.Video.Ffmpeg;

/// <summary>The audio track of an encode.</summary>
/// <param name="EncoderName">The FFmpeg encoder (<c>aac</c>, <c>libopus</c>, ...).</param>
/// <param name="BitRate">Bits per second, or 0 for the encoder's default.</param>
internal sealed record FfmpegAudioPlan(string EncoderName, long BitRate);
