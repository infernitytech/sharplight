using Sharplight.Audio;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Audio;
using Sharplight.Studio.Tests.Fakes;
using Sharplight.Studio.ViewModels.Settings;

namespace Sharplight.Studio.Tests;

public sealed class EngineAudioTests
{
    private static readonly IAudioDecoder s_decoder = new WavAudioDecoder();
    private static readonly IAudioEncoder s_encoder = new WavAudioEncoder();

    [Fact]
    public async Task Player_DecodesTheFile_AndPlaysItOnTheChosenDevice()
    {
        using var services = new TestServices(new AppSettings { AudioOutputDevice = "Headphones" });
        string path = services.Directory + "/tone.wav";
        WavFile.Write(path, new AudioBuffer(new float[8000], 16000, 1));
        var player = new EngineAudioPlayer(s_decoder, services.AudioDevices, services.Settings);

        Assert.True(player.IsAvailable);
        await player.PlayAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal("Headphones", services.AudioDevices.LastOptions!.DeviceName);
        Assert.Equal(16000, services.AudioDevices.LastOptions.SampleRate);
        Assert.Equal(8000, services.AudioDevices.Output!.Written.Count);
        Assert.True(services.AudioDevices.Output.Disposed);
    }

    [Fact]
    public async Task Player_FallsBackToTheDefaultDevice_WhenTheChosenOneIsGone()
    {
        using var services = new TestServices(new AppSettings { AudioOutputDevice = "Unplugged USB DAC" });
        string path = services.Directory + "/tone.wav";
        WavFile.Write(path, new AudioBuffer(new float[800], 16000, 1));
        var player = new EngineAudioPlayer(s_decoder, services.AudioDevices, services.Settings);

        await player.PlayAsync(path, TestContext.Current.CancellationToken);

        Assert.Null(services.AudioDevices.LastOptions!.DeviceName);
    }

    [Fact]
    public async Task PlayerAndRecorder_ReportMissingAudioSupport()
    {
        using var services = new TestServices();
        services.AudioDevices.IsAvailable = false;
        var player = new EngineAudioPlayer(s_decoder, services.AudioDevices, services.Settings);
        var recorder = new EngineAudioRecorder(services.AudioDevices, s_encoder, services.Settings);

        Assert.False(player.IsAvailable);
        Assert.Contains("No audio support", player.Description, StringComparison.Ordinal);
        Assert.False(recorder.IsAvailable);
        Assert.Contains("choose an audio file", recorder.Description, StringComparison.Ordinal);
        await Assert.ThrowsAsync<NotSupportedException>(() => player.PlayAsync("x.wav", TestContext.Current.CancellationToken));
        Assert.Throws<NotSupportedException>(() => recorder.Start("x.wav"));
    }

    [Fact]
    public async Task Recorder_WritesTheCapturedSamplesAsAWavFile()
    {
        using var services = new TestServices(new AppSettings { AudioInputDevice = "Headset" });
        services.AudioDevices.Microphone = Enumerable.Range(0, 3200).Select(i => (i % 100) / 200f).ToArray();
        var recorder = new EngineAudioRecorder(services.AudioDevices, s_encoder, services.Settings);
        string path = services.Directory + "/recording.wav";

        IRecording recording = recorder.Start(path);
        await Task.Delay(100, TestContext.Current.CancellationToken);
        await recording.StopAsync();
        recording.Dispose();

        Assert.Equal("Headset", services.AudioDevices.LastOptions!.DeviceName);
        AudioBuffer written = WavFile.Read(path);
        Assert.Equal(16000, written.SampleRate);
        Assert.Equal(1, written.Channels);
        Assert.Equal(3200, written.FrameCount);
    }

    [Fact]
    public void Settings_ListTheDevices_AndSaveTheChoice()
    {
        using var services = new TestServices(new AppSettings { AudioInputDevice = "Headset" });
        var page = new SettingsPageViewModel(services.Services);

        Assert.Equal(["System default", "Speakers (default)", "Headphones"], page.OutputDevices.Select(d => d.Title));
        Assert.Equal("Headset", page.SelectedInputDevice.Name);
        Assert.Null(page.SelectedOutputDevice.Name);

        page.SelectedOutputDevice = page.OutputDevices[2];

        Assert.Equal("Headphones", services.Settings.Current.AudioOutputDevice);
        Assert.Equal("Headset", services.Settings.Current.AudioInputDevice);
        Assert.Contains("Fake audio", page.AudioStatus, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Transcribe_AcceptsEveryFormatTheNativeDecoderReads()
    {
        Assert.SkipUnless(Sharplight.Audio.Native.NativeAudioLibrary.IsAvailable, "native audio library not built");
        using var services = new TestServices(decoder: new NativeAudioDecoder());
        var page = new Sharplight.Studio.ViewModels.Speech.TranscribePageViewModel(services.Services, new FakeNavigator());
        var audio = new AudioBuffer(new float[48000], 48000, 1);
        string opus = services.Directory + "/voice.opus";
        new NativeAudioEncoder().Save(audio, opus, new AudioEncodeOptions { Format = AudioEncodingFormat.OggOpus });

        Assert.True(await page.SetAudioAsync(opus));
        Assert.True(page.HasAudio);
        Assert.Contains("kHz", page.AudioSummary, StringComparison.Ordinal);
    }
}
