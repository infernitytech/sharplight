using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Audio;
using Sharplight.Audio.Native;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the native audio services of Sharplight.Audio.</summary>
public static class NativeAudioServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds <see cref="IAudioDecoder"/> (WAV, FLAC, MP3, Ogg Vorbis, Ogg Opus), <see cref="IAudioEncoder"/> (WAV, Ogg Opus) and
        /// <see cref="IAudioDeviceProvider"/> (playback and capture) over the native library. When the native library cannot be loaded on
        /// this platform, the decoder and encoder fall back to the built-in WAV implementations and the device provider reports
        /// <see cref="IAudioDeviceProvider.IsAvailable"/> false. Replaces the services of <c>AddWavAudio()</c>; idempotent.
        /// </summary>
        /// <param name="configure">Optional settings of the device provider.</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightAudio(Action<NativeAudioDeviceOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            services.AddOptions();
            if (configure is not null)
            {
                services.Configure(configure);
            }

            services.AddWavAudio();
            if (!services.Any(d => d.ServiceType == typeof(NativeAudioRegistration)))
            {
                services.AddSingleton<NativeAudioRegistration>();
                services.RemoveAll<IAudioDecoder>();
                services.RemoveAll<IAudioEncoder>();
                services.AddSingleton<IAudioDecoder>(_ => NativeAudioLibrary.IsAvailable ? new NativeAudioDecoder() : new WavAudioDecoder());
                services.AddSingleton<IAudioEncoder>(_ => NativeAudioLibrary.IsAvailable ? new NativeAudioEncoder() : new WavAudioEncoder());
            }

            services.TryAddSingleton<NativeAudioDeviceProvider>();
            services.TryAddSingleton<IAudioDeviceProvider>(provider => provider.GetRequiredService<NativeAudioDeviceProvider>());
            return services;
        }
    }
}
