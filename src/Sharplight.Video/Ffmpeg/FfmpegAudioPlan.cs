namespace Sharplight.Video.Ffmpeg;

/// <summary>The audio track of an encode.</summary>
/// <param name="EncoderName">The FFmpeg encoder (<c>aac</c>, <c>libopus</c>, ...).</param>
/// <param name="BitRate">Bits per second, or 0 for the encoder's default.</param>
/// <param name="CodecRequested">The options named the codec, so a source track is transcoded even when the muxer could store it as it is.</param>
internal sealed record FfmpegAudioPlan(string EncoderName, long BitRate, bool CodecRequested);
