using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;
using Sharplight.Video;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the video encoders and the decoder of Sharplight.Video.</summary>
public static class SharplightVideoServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the bundled FFmpeg encoders (<see cref="FfmpegVideoEncoder"/>: MP4, WebM, MOV, Matroska, GIF, APNG) and decoder
        /// (<see cref="FfmpegVideoDecoder"/>) next to the managed APNG and GIF encoders of <c>AddVideoEncoders()</c>, which this method also
        /// adds. The FFmpeg encoders take precedence; where the bundled libraries are missing they are unavailable and the catalog falls back
        /// to the managed encoders. Resolve <see cref="VideoEncoderCatalog"/> / <see cref="VideoDecoderCatalog"/> to pick by format or file.
        /// Idempotent.
        /// </summary>
        /// <param name="configure">Optional settings (hardware encoders).</param>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSharplightVideo(Action<SharplightVideoOptions>? configure = null)
        {
            ArgumentNullException.ThrowIfNull(services);
            services.AddOptions();
            if (configure is not null)
            {
                services.Configure(configure);
            }

            services.AddVideoEncoders();
            if (services.Any(d => d.ImplementationType == typeof(FfmpegVideoDecoder)))
            {
                return services;
            }

            services.TryAddEnumerable(ServiceDescriptor.Singleton<IVideoDecoder, FfmpegVideoDecoder>());
            foreach (string format in FfmpegVideoEncoder.Formats)
            {
                services.AddSingleton<IVideoEncoder>(provider => new FfmpegVideoEncoder(format, provider.GetService<IOptions<SharplightVideoOptions>>()));
            }

            return services;
        }
    }
}
