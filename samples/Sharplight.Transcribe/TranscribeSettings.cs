using Sharplight.Speech;

namespace Sharplight.Transcribe;

/// <summary>The parsed command line of <c>sharplight-transcribe</c>.</summary>
/// <param name="ModelPath">The model: an <c>.slm</c> package, a Hugging Face directory or a whisper.cpp <c>.bin</c> file.</param>
/// <param name="AudioPath">The WAV file to transcribe.</param>
/// <param name="UseCpu">Run on the CPU instead of a GPU.</param>
/// <param name="DeviceIndex">The device index within the provider.</param>
/// <param name="Recognition">The request settings.</param>
/// <param name="Format">The output format.</param>
/// <param name="Verbose">Print load and compile messages to standard error.</param>
public sealed record TranscribeSettings(string ModelPath, string AudioPath, bool UseCpu, int DeviceIndex, SpeechRecognitionOptions Recognition,
    TranscriptFormat Format, bool Verbose);
