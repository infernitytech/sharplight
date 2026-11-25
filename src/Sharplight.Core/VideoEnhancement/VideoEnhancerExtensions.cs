using Sharplight.Vision;

namespace Sharplight.VideoEnhancement;

/// <summary>Convenience calls on <see cref="IVideoEnhancer"/>.</summary>
public static class VideoEnhancerExtensions
{
    extension(IVideoEnhancer enhancer)
    {
        /// <summary>Enhances one still image (a clip of one frame).</summary>
        /// <param name="image">The image.</param>
        /// <param name="options">Output size, seed, colour correction, ..., or <see langword="null"/> for the model's defaults.</param>
        /// <param name="cancellationToken">Stops the request between stages.</param>
        /// <returns>The enhanced image.</returns>
        public async Task<ImageInput> EnhanceImageAsync(ImageInput image, VideoEnhancementOptions? options = null, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(enhancer);
            var result = await enhancer.EnhanceAsync(VideoEnhancementRequest.FromImage(image), options, cancellationToken).ConfigureAwait(false);
            return result.Frames[0];
        }
    }
}
