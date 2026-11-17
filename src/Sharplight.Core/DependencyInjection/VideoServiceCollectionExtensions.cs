using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Video;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the video encoding services.</summary>
public static class VideoServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the <see cref="VideoEncoderCatalog"/> with the built-in managed encoders (animated PNG and GIF) and the (empty)
        /// <see cref="VideoDecoderCatalog"/>. The <c>Sharplight.Video</c> package adds the bundled FFmpeg encoders (MP4, WebM, MOV, Matroska,
        /// and GIF / APNG ahead of the managed ones) and the decoder through <c>AddSharplightVideo()</c>. Idempotent; <c>AddSharplightTasks()</c> calls it.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddVideoEncoders()
        {
            ArgumentNullException.ThrowIfNull(services);
            services.TryAddSingleton<VideoEncoderCatalog>();
            services.TryAddSingleton<VideoDecoderCatalog>();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IVideoEncoder, ApngVideoEncoder>());
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IVideoEncoder, GifVideoEncoder>());
            return services;
        }
    }
}
