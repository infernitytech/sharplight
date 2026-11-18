using System.Globalization;
using System.Text.RegularExpressions;
using Sharplight.Audio;
using Sharplight.Models;
using Sharplight.Studio.ViewModels.Inference.Parameters;
using Sharplight.Studio.ViewModels.Speech;

namespace Sharplight.Studio.Tests;

/// <summary>
/// The Transcribe and Speak pages end to end on real small models (skipped when a model is missing), driven through their view models on
/// the real engine runtime and the fake audio device: timestamps and subtitle files of a transcript, voices, streaming playback and the
/// WAV export of the speech models.
/// </summary>
[Collection(LargeModelCollection.Name)]
public sealed partial class StudioSpeechPagesRealTests
{
    private const string Quote = "ask not what your country can do for you";

    [GeneratedRegex(@"^(\d+)\n(\d\d):(\d\d):(\d\d),(\d{3}) --> (\d\d):(\d\d):(\d\d),(\d{3})\n(.+)$", RegexOptions.Multiline)]
    private static partial Regex SrtCue();

    [Theory]
    [InlineData("whisper.cpp/ggml-tiny.en.bin", "whisper-tiny")]
    [InlineData("hf/Qwen3-ASR-0.6B", "qwen3-asr")]
    [InlineData("hf/parakeet-ctc-0.6b", "parakeet")]
    public async Task Transcribe_GivesTimedSegmentsOfJfk_AndExportsSrtAndVtt(string relative, string id)
    {
        string path = RealStudioSession.Require(relative);
        string jfk = RealStudioSession.Require("audio/jfk.wav");
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        await session.LoadAsync(id, path, ModelKinds.SpeechRecognition, ct);

        var page = new TranscribePageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        Assert.Equal(id, page.SelectedModel?.Id);
        Assert.True(await page.SetAudioAsync(jfk));
        page.RunCommand.Execute(null);
        await RealStudioSession.FinishAsync(page);

        Assert.Contains(Quote, page.TranscriptText, StringComparison.OrdinalIgnoreCase);
        Assert.NotEmpty(page.Segments);
        double duration = WavFile.Read(jfk).Duration.TotalSeconds;
        double previousEnd = 0;
        foreach (var segment in page.Segments)
        {
            var record = segment.Record;
            Assert.True(record.Start >= 0 && record.End > record.Start, $"{record.Start} - {record.End}");
            Assert.True(record.Start >= previousEnd - 0.5, "segments are in order");
            Assert.True(record.End <= duration + 1.0, $"segment ends at {record.End} s of a {duration:0.0} s file");
            previousEnd = record.End;
            Assert.NotEmpty(segment.Text);
        }

        Assert.True(page.Segments[^1].Record.End > duration / 2, "the segments cover the speech");

        // The exports: what the page saves where the dialog says, and what the run keeps in the gallery.
        string srtPath = Path.Combine(session.Directory, "jfk.srt");
        string vttPath = Path.Combine(session.Directory, "jfk.vtt");
        session.Platform.SaveTo = srtPath;
        await page.ExportCommand.ExecuteAsync("srt");
        session.Platform.SaveTo = vttPath;
        await page.ExportCommand.ExecuteAsync("vtt");

        string srt = await File.ReadAllTextAsync(srtPath, ct);
        var cues = SrtCue().Matches(srt);
        Assert.Equal(page.Segments.Count, cues.Count);
        for (int i = 0; i < cues.Count; i++)
        {
            Assert.Equal((i + 1).ToString(CultureInfo.InvariantCulture), cues[i].Groups[1].Value);
            double start = Seconds(cues[i], 2);
            double end = Seconds(cues[i], 6);
            Assert.Equal(page.Segments[i].Record.Start, start, 0.002);
            Assert.Equal(page.Segments[i].Record.End, end, 0.002);
            Assert.Equal(page.Segments[i].Text, cues[i].Groups[10].Value.Trim());
        }

        string vtt = await File.ReadAllTextAsync(vttPath, ct);
        Assert.StartsWith("WEBVTT\n\n", vtt, StringComparison.Ordinal);
        Assert.Equal(cues.Count, Regex.Matches(vtt, @"\d\d:\d\d:\d\d\.\d{3} --> \d\d:\d\d:\d\d\.\d{3}").Count);
        Assert.Contains(Quote, vtt.Replace("\n", " ", StringComparison.Ordinal), StringComparison.OrdinalIgnoreCase);

        var output = Assert.Single(page.Gallery.Items);
        Assert.Equal(srt, await File.ReadAllTextAsync(output.Files.Single(file => file.Name == "transcript.srt").Path, ct));
        Assert.Contains("speed:", page.ResultSummary, StringComparison.Ordinal);
    }

    private static double Seconds(Match cue, int firstGroup)
        => int.Parse(cue.Groups[firstGroup].Value, CultureInfo.InvariantCulture) * 3600
           + int.Parse(cue.Groups[firstGroup + 1].Value, CultureInfo.InvariantCulture) * 60
           + int.Parse(cue.Groups[firstGroup + 2].Value, CultureInfo.InvariantCulture)
           + int.Parse(cue.Groups[firstGroup + 3].Value, CultureInfo.InvariantCulture) / 1000.0;

    [Fact]
    public async Task Transcribe_WithoutTimestamps_StillGivesTheText()
    {
        string whisper = RealStudioSession.Require("whisper.cpp/ggml-tiny.en.bin");
        string jfk = RealStudioSession.Require("audio/jfk.wav");
        using var session = new RealStudioSession();
        await session.LoadAsync("whisper-tiny", whisper, ModelKinds.SpeechRecognition, TestContext.Current.CancellationToken);
        var page = new TranscribePageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        page.Panel.Apply(new Dictionary<string, string> { ["timestamps"] = "none" });
        Assert.True(await page.SetAudioAsync(jfk));
        page.RunCommand.Execute(null);
        await RealStudioSession.FinishAsync(page);
        Assert.Contains(Quote, page.TranscriptText, StringComparison.OrdinalIgnoreCase);
    }

    [Theory]
    [InlineData("tts/kokoro", "kokoro", "af_heart", 24000)]
    [InlineData("tts/Qwen3-TTS-12Hz-0.6B-CustomVoice", "qwen3-tts", "", 24000)]
    public async Task Speak_OffersTheVoices_PlaysWhileGenerating_AndExportsAWav(string relative, string id, string voice, int sampleRate)
    {
        string path = RealStudioSession.Require(relative);
        using var session = new RealStudioSession();
        var ct = TestContext.Current.CancellationToken;
        var model = await session.LoadAsync(id, path, ModelKinds.SpeechSynthesis, ct);

        var page = new SpeakPageViewModel(session.Services, new Fakes.FakeNavigator());
        await page.ActivateAsync();
        await page.RefreshVoicesAsync(model);
        var voices = page.Panel.Parameters.OfType<ChoiceParameterViewModel>().Single(parameter => parameter.Key == "voice");
        Assert.True(voices.Choices.Count > 2, string.Join(", ", voices.Choices.Select(choice => choice.Value)));
        string chosen = voice.Length > 0 ? voice : voices.Choices[^1].Value;
        Assert.Contains(voices.Choices, choice => choice.Value == chosen);

        // Playing while generating: the chunks reach the fake speaker through the player while the run is still going.
        page.Panel.Apply(new Dictionary<string, string> { ["voice"] = chosen, ["stream"] = "true" });
        page.InputText = "Hello world. This is Sharplight speaking.";
        page.RunCommand.Execute(null);
        await RealStudioSession.FinishAsync(page);
        await RealStudioSession.WaitFor(() => !page.IsPlaying);
        Assert.Equal(sampleRate, session.Audio.PlayedSampleRate);
        Assert.True(session.Audio.Played.Count > sampleRate, $"{session.Audio.Played.Count} samples played");
        Assert.True(RealStudioSession.IsAudible(session.Audio.Played));
        Assert.True(page.HasResult);
        Assert.True(string.IsNullOrEmpty(page.Notice), page.Notice);

        // The WAV export is the whole speech, the same length as what was played.
        string target = Path.Combine(session.Directory, "speech.wav");
        session.Platform.SaveTo = target;
        await page.SaveCommand.ExecuteAsync(null);
        var exported = WavFile.Read(target);
        Assert.Equal(sampleRate, exported.SampleRate);
        Assert.True(exported.Duration.TotalSeconds > 1.5, $"{exported.Duration.TotalSeconds:0.0} s");
        Assert.True(RealStudioSession.IsAudible(exported.Samples.ToArray()));
        Assert.Equal(session.Audio.Played.Count, exported.Samples.Length, session.Audio.Played.Count / 20.0);
        Assert.Single(page.Gallery.Items);
    }
}
