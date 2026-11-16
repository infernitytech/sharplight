using Microsoft.Extensions.DependencyInjection;

namespace Sharplight.Audio.Tests;

public sealed class RegistrationTests
{
    [Fact]
    public void AddSharplightAudioReplacesTheWavFallback()
    {
        var services = new ServiceCollection();
        services.AddWavAudio();
        services.AddSharplightAudio(o => o.UseNullBackend = true);
        services.AddSharplightAudio();
        using ServiceProvider provider = services.BuildServiceProvider();

        if (Native.NativeAudioLibrary.IsAvailable)
        {
            Assert.IsType<NativeAudioDecoder>(provider.GetRequiredService<IAudioDecoder>());
            Assert.IsType<NativeAudioEncoder>(provider.GetRequiredService<IAudioEncoder>());
        }
        else
        {
            Assert.IsType<WavAudioDecoder>(provider.GetRequiredService<IAudioDecoder>());
        }

        Assert.Single(services, d => d.ServiceType == typeof(IAudioDecoder));
        Assert.Same(provider.GetRequiredService<IAudioDeviceProvider>(), provider.GetRequiredService<IAudioDeviceProvider>());
        Assert.IsType<NativeAudioDeviceProvider>(provider.GetRequiredService<IAudioDeviceProvider>());
    }

    [Fact]
    public void WavFallbackWorksWithoutNativeLibrary()
    {
        var services = new ServiceCollection();
        services.AddWavAudio();
        using ServiceProvider provider = services.BuildServiceProvider();
        var audio = new AudioBuffer([0f, 0.5f, -0.5f, 0.25f], 8000, 1);
        IAudioEncoder encoder = provider.GetRequiredService<IAudioEncoder>();
        Assert.False(encoder.SupportsFormat(AudioEncodingFormat.OggOpus));
        AudioBuffer decoded = provider.GetRequiredService<IAudioDecoder>().Decode(encoder.Encode(audio, new AudioEncodeOptions { Format = AudioEncodingFormat.WavFloat32 }));
        Assert.Equal(audio.Samples, decoded.Samples);
        Assert.Throws<NotSupportedException>(() => encoder.Encode(audio, new AudioEncodeOptions { Format = AudioEncodingFormat.OggOpus }));
    }
}
