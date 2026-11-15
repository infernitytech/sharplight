using Sharplight.Models;

namespace Sharplight.Architectures;

/// <summary>
/// Registers the architecture adapters shipped with Sharplight (decoder, encoder, hybrid and multimodal families) with
/// <see cref="ModelAdapterRegistry"/> and their multimodal handlers with <see cref="Sharplight.Vision.MultimodalRegistry"/>. The core
/// loaders (<c>LlmModel</c>, <c>EmbeddingModel</c>, <c>TaskModel</c>) only consult the registries, so a host calls this once at
/// startup, directly or through <c>services.AddSharplight()</c> of the <c>Sharplight.Defaults</c> package (which also registers the
/// importers and execution providers).
/// </summary>
public static class BuiltinAdapters
{
    private static int s_done;

    /// <summary>Registers all built-in adapters. Idempotent and thread-safe.</summary>
    public static void RegisterAll()
    {
        if (Interlocked.Exchange(ref s_done, 1) == 1)
        {
            return;
        }

        ModelAdapterRegistry.Register(new Qwen35Adapter());
        Sharplight.Vision.MultimodalRegistry.Register(new Qwen35Multimodal());
        ModelAdapterRegistry.Register(new DecoderAdapter());
        ModelAdapterRegistry.Register(new EncoderAdapter());
        ModelAdapterRegistry.Register(new ImageEncoderAdapter());
        ModelAdapterRegistry.Register(new ImageTextAdapter());
        ModelAdapterRegistry.Register(new DetrAdapter());
        ModelAdapterRegistry.Register(new EncoderDecoderAdapter());
        ModelAdapterRegistry.Register(new NemotronHAdapter());
        ModelAdapterRegistry.Register(new HybridAdapter());
        ModelAdapterRegistry.Register(new DepthMemoryAdapter());
        ModelAdapterRegistry.Register(new Rwkv7Adapter());
        ModelAdapterRegistry.Register(new Rwkv6Adapter());
        ModelAdapterRegistry.Register(new LatentDecoderAdapter());
        ModelAdapterRegistry.RegisterComposite(new OuteTtsAdapter());
        ModelAdapterRegistry.RegisterAuxiliaryOnly(WavTokenizerParameters.Architecture, Sharplight.Vision.AuxiliaryRoles.Vocoder);
        ModelAdapterRegistry.RegisterAuxiliaryOnly(Eagle3DraftHead.Architecture, Sharplight.Vision.AuxiliaryRoles.Draft);
        ModelAdapterRegistry.RegisterAuxiliaryOnly(SharedKvDraftHead.Architecture, Sharplight.Vision.AuxiliaryRoles.Draft);
        ModelAdapterRegistry.RegisterAuxiliaryOnly(DflashDraftHead.Architecture, Sharplight.Vision.AuxiliaryRoles.Draft);
    }
}
