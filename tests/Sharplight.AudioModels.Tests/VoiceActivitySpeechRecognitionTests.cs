using Microsoft.Extensions.DependencyInjection;
using Sharplight.Audio;
using Sharplight.Speech;
using Sharplight.Tasks;

namespace Sharplight.AudioModels.Tests;

/// <summary>Voice activity detection as the pre-segmentation of long-form speech recognition.</summary>
public class VoiceActivitySpeechRecognitionTests
{
    [Fact]
    public async Task Windows_group_speech_and_keep_the_recording_timeline()
    {
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string wav = Path.Combine(home, "models", "audio", "librispeech-1272-128104-0004.wav");
        string recognizerPath = Path.Combine(home, "models", "hf", "moonshine-tiny");
        AudioModels.Require(AudioModels.Model("silero-vad/silero_vad_16k_op15.onnx"), wav, recognizerPath);
        using var vadModel = AudioModels.Load(AudioModels.Open(gpu: false), AudioModels.Model("silero-vad/silero_vad_16k_op15.onnx"));
        using var detector = AudioModels.Client<IVoiceActivityDetector>(vadModel);
        using var asrModel = AudioModels.Load(AudioModels.Open(gpu: false), recognizerPath);
        using var recognizer = AudioModels.Services.GetRequiredService<TaskClientProvider>().Create<ISpeechRecognizer>(asrModel);
        var audio = WavFile.Read(wav);

        var plain = await recognizer.TranscribeAsync(audio);
        var segmented = await recognizer.TranscribeAsync(audio, new SpeechRecognitionOptions { UseVoiceActivityDetection = true, VoiceActivityDetector = detector });

        Assert.NotEmpty(segmented.Segments);
        Assert.True(segmented.Segments[^1].End <= audio.Duration + TimeSpan.FromSeconds(1));
        Assert.True(segmented.Segments.Zip(segmented.Segments.Skip(1), (a, b) => a.Start <= b.Start).All(ordered => ordered));
        var words = (string text) => text.ToLowerInvariant().Split([' ', ',', '.', '?', '!'], StringSplitOptions.RemoveEmptyEntries).ToHashSet();
        var reference = words(plain.Text);
        double shared = words(segmented.Text).Count(reference.Contains) / (double)reference.Count;
        Assert.True(shared > 0.8, $"The pre-segmented transcript shares {shared:P0} of the plain transcript's words.");
        await Assert.ThrowsAsync<InvalidOperationException>(() => recognizer.TranscribeAsync(audio, new SpeechRecognitionOptions { UseVoiceActivityDetection = true }));
    }
}
