using Sharplight.Architectures;
using Sharplight.Vision;

namespace Microsoft.Extensions.DependencyInjection;

/// <summary>Registers the architecture adapters shipped with Sharplight.</summary>
public static class ArchitectureServiceCollectionExtensions
{
    extension(IServiceCollection services)
    {
        /// <summary>
        /// Adds the built-in architecture adapters (decoder, encoder, encoder-decoder, hybrid, recurrent, latent-attention, image-encoder, image-text, detection, depth, latent-diffusion and
        /// multimodal families), the multimodal prompt assembly of the Qwen-VL family, the text-to-speech families (OuteTTS with the WavTokenizer
        /// decoder as its <see cref="AuxiliaryRoles.Vocoder"/> source, and its <see cref="Sharplight.Speech.ISpeechSynthesizer"/> factory),
        /// and the drafter architectures that only load as the <see cref="AuxiliaryRoles.Draft"/> source of another model. Idempotent. <c>services.AddSharplight()</c> of the
        /// Sharplight package calls it.
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
                .AddModelAdapter<ConformerAsrAdapter>()
                .AddModelAdapter<Wav2Vec2CtcAdapter>()
                .AddModelAdapter<NemotronHAdapter>()
                .AddModelAdapter<HybridAdapter>()
                .AddModelAdapter<DepthMemoryAdapter>()
                .AddModelAdapter<Rwkv7Adapter>()
                .AddModelAdapter<Rwkv6Adapter>()
                .AddModelAdapter<LatentDecoderAdapter>()
                .AddModelAdapter<ImageEncoderAdapter>()
                .AddModelAdapter<ImageTextAdapter>()
                .AddModelAdapter<DetrAdapter>()
                .AddModelAdapter<RtDetrAdapter>()
                .AddModelAdapter<DepthEstimationAdapter>()
                .AddModelAdapter<SegformerAdapter>()
                .AddModelAdapter<BiRefNetAdapter>()
                .AddModelAdapter<Sam2Adapter>()
                .AddModelAdapter<RrdbNetAdapter>()
                .AddModelAdapter<SwinIrAdapter>()
                .AddModelAdapter<ResNetClassifierAdapter>()
                .AddModelAdapter<KokoroAdapter>()
                .AddModelAdapter<EncodecAdapter>()
                .AddModelAdapter<DacAdapter>()
                .AddModelAdapter<MimiAdapter>()
                .AddModelAdapter<SileroVadAdapter>()
                .AddModelAdapter<ClapAdapter>()
                .AddModelAdapter<MusicGenAdapter>()
                .AddModelAdapter<DemucsAdapter>()
                .AddTaskClientFactory<KokoroSynthesizerFactory>()
                .AddCompositeModelAdapter<OrpheusAdapter>()
                .AddTaskClientFactory<OrpheusSynthesizerFactory>()
                .AddModelAdapter<ChatterboxAdapter>()
                .AddTaskClientFactory<ChatterboxSynthesizerFactory>()
                .AddCompositeModelAdapter<F5Adapter>()
                .AddAuxiliaryArchitecture("vocos", AuxiliaryRoles.Vocoder)
                .AddTaskClientFactory<F5SynthesizerFactory>()
                .AddModelAdapter<XttsAdapter>()
                .AddTaskClientFactory<XttsSynthesizerFactory>()
                .AddAuxiliaryArchitecture(SnacParameters.Architecture, AuxiliaryRoles.Vocoder)
                .AddModelAdapter<LatentDiffusionAdapter>()
                .AddModelAdapter<FluxAdapter>()
                .AddModelAdapter<WanAdapter>()
                .AddCompositeModelAdapter<WanCompanionAdapter>()
                .AddCompositeModelAdapter<HunyuanVideoCompanionAdapter>()
                .AddCompositeModelAdapter<LtxCompanionAdapter>()
                .AddModelAdapter<LtxAdapter>()
                .AddModelAdapter<HunyuanVideoAdapter>()
                .AddCompositeModelAdapter<FluxCompanionAdapter>()
                .AddModelAdapter<Sd3Adapter>()
                .AddCompositeModelAdapter<Sd3CompanionAdapter>()
                .AddModelAdapter<ChromaAdapter>()
                .AddCompositeModelAdapter<ChromaCompanionAdapter>()
                .AddModelAdapter<ZImageAdapter>()
                .AddCompositeModelAdapter<ZImageCompanionAdapter>()
                .AddModelAdapter<QwenImageAdapter>()
                .AddCompositeModelAdapter<QwenImageCompanionAdapter>()
                .AddModelAdapter<QwenImage21Adapter>()
                .AddCompositeModelAdapter<QwenImage21CompanionAdapter>()
                .AddModelAdapter<Flux2Adapter>()
                .AddCompositeModelAdapter<Flux2CompanionAdapter>()
                .AddAuxiliaryArchitecture(DiffusionCompanionAdapter.ClipTextArchitecture, AuxiliaryRoles.TextEncoder)
                .AddAuxiliaryArchitecture(DiffusionCompanionAdapter.AutoencoderArchitecture, AuxiliaryRoles.Vae)
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
