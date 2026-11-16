using Microsoft.Extensions.DependencyInjection.Extensions;
using Sharplight.Video;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the video encoding services.</summary>
public static class VideoServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the <see cref="VideoEncoderCatalog"/> and the built-in managed encoders (animated PNG and GIF). The <c>Sharplight.Video</c>
        /// package adds WebM and MP4 through <c>AddSharplightVideo()</c>. Idempotent; <c>AddSharplightTasks()</c> calls it.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddVideoEncoders()
        {
            ArgumentNullException.ThrowIfNull(services);
            services.TryAddSingleton<VideoEncoderCatalog>();
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IVideoEncoder, ApngVideoEncoder>());
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IVideoEncoder, GifVideoEncoder>());
            return services;
        }
    }
}
