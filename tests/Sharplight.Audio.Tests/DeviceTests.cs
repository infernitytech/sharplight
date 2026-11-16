using Microsoft.Extensions.Options;

namespace Sharplight.Audio.Tests;

public sealed class DeviceTests
{
    private static NativeAudioDeviceProvider NullProvider() => new(Options.Create(new NativeAudioDeviceOptions { UseNullBackend = true }));

    [Fact]
    public async Task NullBackendEnumeratesAndPlays()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        using NativeAudioDeviceProvider provider = NullProvider();
        Assert.True(provider.IsAvailable);
        Assert.Equal("Null", provider.Description);
        Assert.NotEmpty(provider.GetDevices(AudioDeviceKind.Playback));
        Assert.NotEmpty(provider.GetDevices(AudioDeviceKind.Capture));

        using IAudioOutput output = provider.OpenOutput(new AudioStreamOptions { SampleRate = 48000, Channels = 2, BufferDuration = TimeSpan.FromMilliseconds(200) });
        var tone = new AudioBuffer(TestSignals.Tone(48000, 2, 4800), 48000, 2);
        await output.PlayAsync(tone, TestContext.Current.CancellationToken);
        Assert.Equal(0, output.QueuedFrames);
        Assert.Equal(4800, output.Statistics.FramesTransferred);
    }

    [Fact]
    public async Task PlayAsyncConvertsAndCancels()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        using NativeAudioDeviceProvider provider = NullProvider();
        using IAudioOutput output = provider.OpenOutput(new AudioStreamOptions { SampleRate = 48000, Channels = 2 });
        var mono16k = new AudioBuffer(TestSignals.Tone(16000, 1, 1600), 16000, 1);
        await output.PlayAsync(mono16k, TestContext.Current.CancellationToken);
        Assert.Equal(4800, output.Statistics.FramesTransferred);

        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));
        var longTone = new AudioBuffer(TestSignals.Tone(48000, 2, 48000 * 20), 48000, 2);
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => output.PlayAsync(longTone, cancellation.Token));
    }

    [Fact]
    public async Task NullCaptureDeliversFrames()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        using NativeAudioDeviceProvider provider = NullProvider();
        using IAudioInput input = provider.OpenInput(new AudioStreamOptions { SampleRate = 16000, Channels = 1 });
        input.Start();
        await Task.Delay(500, TestContext.Current.CancellationToken);
        var buffer = new float[16000];
        int frames = input.Read(buffer);
        Assert.True(frames > 0);
        Assert.True(input.Statistics.FramesTransferred >= frames);
    }

    [Fact]
    public void UnknownDeviceNameIsReported()
    {
        Assert.SkipUnless(Native.NativeAudioLibrary.IsAvailable, "native library not built");
        using NativeAudioDeviceProvider provider = NullProvider();
        Assert.Throws<InvalidOperationException>(() => provider.OpenOutput(new AudioStreamOptions { DeviceName = "no such device" }));
    }

    [Fact]
    public void MachineWithoutAudioIsHandledGracefully()
    {
        using var provider = new NativeAudioDeviceProvider();
        if (provider.IsAvailable)
        {
            Assert.NotEmpty(provider.Description);
            return;
        }

        Assert.NotEmpty(provider.Description);
        Assert.Empty(provider.GetDevices(AudioDeviceKind.Playback));
        Assert.Throws<InvalidOperationException>(() => provider.OpenOutput());
        Assert.Throws<InvalidOperationException>(() => provider.OpenInput());
    }
}
