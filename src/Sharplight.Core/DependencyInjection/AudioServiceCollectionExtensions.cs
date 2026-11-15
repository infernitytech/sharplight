using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Audio;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the built-in audio services (WAV only). The <c>Sharplight.Audio</c> package replaces them with native implementations.</summary>
public static class AudioServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the WAV-only <see cref="IAudioDecoder"/> and <see cref="IAudioEncoder"/> unless an implementation is already registered
        /// (for example by <c>services.AddSharplightAudio()</c>). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddWavAudio()
        {
            ArgumentNullException.ThrowIfNull(services);
            services.TryAddSingleton<IAudioDecoder, WavAudioDecoder>();
            services.TryAddSingleton<IAudioEncoder, WavAudioEncoder>();
            return services;
        }
    }
}
