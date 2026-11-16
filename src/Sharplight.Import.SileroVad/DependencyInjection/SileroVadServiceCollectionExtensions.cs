using Sharplight.Import.SileroVad;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the Silero VAD checkpoint format.</summary>
public static class SileroVadServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>Adds the Silero VAD importer (<see cref="SileroVadImporter"/>), so the model loader opens the ONNX export of Silero VAD as a voice activity model. Idempotent.</summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddSileroVadImporter() => services.AddModelImporter<SileroVadImporter>();
    }
}
