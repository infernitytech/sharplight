using Sharplight.Import.CosyVoice;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the CosyVoice checkpoint format.</summary>
public static class CosyVoiceServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the CosyVoice importer (<see cref="CosyVoiceImporter"/>), so the model loader opens Fun-CosyVoice3 release folders (<c>cosyvoice3.yaml</c>,
        /// <c>llm.pt</c>, <c>flow.pt</c>, <c>hift.pt</c>, <c>CosyVoice-BlankEN/</c>, optionally <c>speech_tokenizer_v3.onnx</c> and <c>campplus.onnx</c>). Idempotent.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddCosyVoiceImporter() => services.AddModelImporter<CosyVoiceImporter>();
    }
}
