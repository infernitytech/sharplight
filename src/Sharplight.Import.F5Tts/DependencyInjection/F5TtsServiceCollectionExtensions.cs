using Sharplight.Import.F5Tts;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the F5-TTS and Vocos checkpoint formats.</summary>
public static class F5TtsServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the F5-TTS importer (<see cref="F5TtsImporter"/>: a folder with <c>model_*.safetensors</c> and <c>vocab.txt</c>) and the Vocos importer
        /// (<see cref="VocosImporter"/>: a folder with <c>config.yaml</c> and <c>pytorch_model.bin</c>, loaded as the F5-TTS model's <c>vocoder</c>). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddF5TtsImporter() => services.AddModelImporter<F5TtsImporter>().AddModelImporter<VocosImporter>();
    }
}
