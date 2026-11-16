using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Video;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the video encoders of Sharplight.Video.</summary>
public static class SharplightVideoServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds <see cref="WebMVideoEncoder"/> (VP8 / VP9 over the native libvpx; unavailable where the native library is missing) and
        /// <see cref="FfmpegVideoEncoder"/> (MP4 through a system ffmpeg; unavailable without it) next to the managed APNG and GIF encoders
        /// of <c>AddVideoEncoders()</c>, which this method also adds. Resolve <see cref="VideoEncoderCatalog"/> to pick by format or file
        /// extension; <see cref="IVideoEncoder.IsAvailable"/> tells what works on this machine. Idempotent.
        /// </summary>
        /// <param name="configure">Optional settings (WebM codec, ffmpeg path).</param>
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
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IVideoEncoder, WebMVideoEncoder>());
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IVideoEncoder, FfmpegVideoEncoder>());
            return services;
        }
    }
}
