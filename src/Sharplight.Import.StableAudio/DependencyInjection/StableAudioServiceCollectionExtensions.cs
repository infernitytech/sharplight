using Sharplight.Import.StableAudio;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the stable-audio-3 checkpoint format.</summary>
public static class StableAudioServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the Stable Audio importer (<see cref="StableAudioImporter"/>), so the model loader opens stable-audio-3 checkpoint folders (<c>model_config.json</c>,
        /// <c>model.safetensors</c> and the T5Gemma text encoder folder). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddStableAudioImporter() => services.AddModelImporter<StableAudioImporter>();
    }
}
