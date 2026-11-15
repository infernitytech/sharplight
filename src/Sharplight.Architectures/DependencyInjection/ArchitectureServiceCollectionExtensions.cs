using Sharplight.Architectures;
using Sharplight.Vision;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the architecture adapters shipped with Sharplight.</summary>
public static class ArchitectureServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the built-in architecture adapters (decoder, encoder, encoder-decoder, hybrid, recurrent, latent-attention and multimodal
        /// families), the multimodal prompt assembly of the Qwen-VL family, the text-to-speech families (OuteTTS with the WavTokenizer
        /// decoder as its <see cref="AuxiliaryRoles.Vocoder"/> source, and its <see cref="Sharplight.Speech.ISpeechSynthesizer"/> factory),
        /// and the drafter architectures that only load as the <see cref="AuxiliaryRoles.Draft"/> source of another model. Idempotent. <c>services.AddSharplight()</c> of the
        /// Sharplight.Defaults package calls it.
        /// </summary>
        /// <returns>The service collection.</returns>
        public IServiceCollection AddBuiltinArchitectures()
        {
            ArgumentNullException.ThrowIfNull(services);
            services.AddModelAdapter<Qwen35Adapter>()
                .AddMultimodalArchitecture<Qwen35Multimodal>()
                .AddModelAdapter<DecoderAdapter>()
                .AddModelAdapter<EncoderAdapter>()
                .AddModelAdapter<EncoderDecoderAdapter>()
                .AddModelAdapter<NemotronHAdapter>()
                .AddModelAdapter<HybridAdapter>()
                .AddModelAdapter<DepthMemoryAdapter>()
                .AddModelAdapter<Rwkv7Adapter>()
                .AddModelAdapter<Rwkv6Adapter>()
                .AddModelAdapter<LatentDecoderAdapter>()
                .AddCompositeModelAdapter<OuteTtsAdapter>()
                .AddTaskClientFactory<OuteTtsSynthesizerFactory>()
                .AddAuxiliaryArchitecture(WavTokenizerParameters.Architecture, AuxiliaryRoles.Vocoder)
                .AddCompositeModelAdapter<OuteTtsAdapter>()
                .AddTaskClientFactory<OuteTtsSynthesizerFactory>()
                .AddAuxiliaryArchitecture(WavTokenizerParameters.Architecture, AuxiliaryRoles.Vocoder)
                .AddAuxiliaryArchitecture(Eagle3DraftHead.Architecture, AuxiliaryRoles.Draft)
                .AddAuxiliaryArchitecture(SharedKvDraftHead.Architecture, AuxiliaryRoles.Draft)
                .AddAuxiliaryArchitecture(DflashDraftHead.Architecture, AuxiliaryRoles.Draft);

            return services;
        }
    }
}
