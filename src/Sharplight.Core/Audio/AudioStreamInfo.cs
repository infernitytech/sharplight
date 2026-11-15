namespace Sharplight.Audio;

/// <summary>Format of an audio file or stream as stored (before any conversion).</summary>
/// <param name="SampleRate">Frames per second.</param>
/// <param name="Channels">Channels per frame.</param>
/// <param name="TotalFrames">Length in frames, or <see langword="null"/> when the container does not tell.</param>
/// <param name="Codec">Lower-case codec name: <c>wav</c>, <c>flac</c>, <c>mp3</c>, <c>vorbis</c>, <c>opus</c>, or <c>unknown</c>.</param>
public sealed record AudioStreamInfo(int SampleRate, int Channels, long? TotalFrames, string Codec);
