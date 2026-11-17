using Sharplight.Audio;
using Sharplight.Llm;
using Sharplight.Speech;

namespace Sharplight.TextToSpeech.Tests;

/// <summary>Saved voices: the library stores and reloads tensors exactly, and a saved voice speaks exactly like the recording it was made from.</summary>
public class VoiceProfileTests
{
    private static string Root => Path.GetDirectoryName(TtsModels.Kokoro)!;

    private static string Recording => Path.Combine(Path.GetDirectoryName(Root)!, "audio", "librispeech-1272-128104-0004.wav");

    [Fact]
    public void Library_round_trips_tensors_and_lists_by_architecture()
    {
        string folder = Path.Combine(Path.GetTempPath(), "sl-voices-" + Guid.NewGuid().ToString("N"));
        try
        {
            var library = new VoiceLibrary(folder);
            Assert.Empty(library.List());
            var voice = new VoiceProfile
            {
                Name = "Anna Müller",
                Architecture = "xtts",
                ReferenceText = "hello there",
                SampleRate = 24000,
                Tensors = new Dictionary<string, VoiceTensor> { ["a"] = VoiceTensor.From([1.5f, -2f, 3f], [3, 1]), ["b"] = VoiceTensor.From([7, 8]) },
            };
            library.Save(voice);
            library.Save(voice.Renamed("Other"));
            library.Save(new VoiceProfile { Name = "Cosy", Architecture = "cosyvoice3", Tensors = new Dictionary<string, VoiceTensor>() });
            Assert.Equal(["Anna Müller", "Cosy", "Other"], library.List());
            Assert.Equal(["Cosy"], library.List("cosyvoice3"));
            var loaded = library.Load("Anna Müller");
            Assert.Equal("xtts", loaded.Architecture);
            Assert.Equal("hello there", loaded.ReferenceText);
            Assert.Equal([1.5f, -2f, 3f], loaded.Tensors["a"].ToFloats());
            Assert.Equal([3, 1], loaded.Tensors["a"].Shape);
            Assert.Equal([7, 8], loaded.Tensors["b"].ToInts());
            library.Rename("Other", "Renamed");
            Assert.True(library.Contains("Renamed"));
            Assert.False(library.Contains("Other"));
            Assert.True(library.Delete("Renamed"));
            Assert.False(library.Delete("Renamed"));
            Assert.Throws<FileNotFoundException>(() => library.Load("Renamed"));
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }

    [Theory]
    [InlineData("XTTS-v2", "mel_stats.pth", true)]
    [InlineData("chatterbox", "ve.safetensors", true)]
    [InlineData("Fun-CosyVoice3-0.5B", "campplus.onnx", true)]
    public async Task A_saved_voice_speaks_like_its_recording(string model, string marker, bool gpu)
    {
        string path = Path.Combine(Root, model);
        TtsModels.Require(Recording);
        TtsModels.Require(Path.Combine(path, marker));

        using var loaded = TtsModels.Load(TtsModels.Open(gpu), path);
        using var synthesizer = TtsModels.Synthesizer(loaded);
        var cloner = Assert.IsAssignableFrom<IVoiceCloner>(synthesizer.GetService(typeof(IVoiceCloner)));
        var recording = WavFile.Read(Recording);
        const string ReferenceText = "A man said to the universe, sir I exist.";
        var sampling = new SamplingOptions { Seed = 7 };
        const string Text = "The weather today is sunny.";
        var viaRecording = new SpeechSynthesisOptions { ReferenceAudio = recording, ReferenceText = ReferenceText, Sampling = sampling };

        var voice = await cloner.CreateVoiceAsync("test", recording, ReferenceText, TestContext.Current.CancellationToken);
        string folder = Path.Combine(Path.GetTempPath(), "sl-voices-" + Guid.NewGuid().ToString("N"));
        try
        {
            var library = new VoiceLibrary(folder);
            library.Save(voice);
            var reloaded = library.Load("test");
            Assert.Equal(cloner.VoiceArchitecture, reloaded.Architecture);
            var expected = (await synthesizer.SynthesizeAsync(Text, viaRecording, TestContext.Current.CancellationToken)).Audio.Samples;
            var actual = (await synthesizer.SynthesizeAsync(Text, new SpeechSynthesisOptions { VoiceProfile = reloaded, Sampling = sampling }, TestContext.Current.CancellationToken))
                .Audio.Samples;
            double bytes = reloaded.Tensors.Sum(t => (double)t.Value.Data.Length);
            ParityReport.Write($"Saved voice {model} {(gpu ? "vulkan" : "cpu")}: {bytes / 1024:F0} KiB of conditioning, {expected.Length} samples via the recording and {actual.Length} via the saved voice, "
                + $"identical {expected.AsSpan().SequenceEqual(actual)}");
            Assert.Equal(expected.Length, actual.Length);
            Assert.True(expected.AsSpan().SequenceEqual(actual), "the saved voice's audio differs from the recording's");
                        Assert.Throws<ArgumentException>(() => synthesizer.SynthesizeAsync(Text, new SpeechSynthesisOptions
            {
                VoiceProfile = new VoiceProfile { Name = "other", Architecture = "nothing", Tensors = voice.Tensors },
            }).GetAwaiter().GetResult());
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }
    }
}
