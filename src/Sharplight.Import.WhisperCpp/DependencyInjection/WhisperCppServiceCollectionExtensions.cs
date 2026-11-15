using Sharplight.Import.WhisperCpp;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the whisper.cpp model format.</summary>
public static class WhisperCppServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the whisper.cpp importer (<see cref="WhisperCppImporter"/>), so the model loader opens whisper.cpp <c>ggml-*.bin</c> files.
        /// Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddWhisperCppImporter() => services.AddModelImporter<WhisperCppImporter>();
    }
}
