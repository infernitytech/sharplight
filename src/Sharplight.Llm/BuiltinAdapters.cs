using Sharplight.Architectures;
using Sharplight.Models;

namespace Sharplight.Llm;

/// <summary>
/// Registers the architecture adapters shipped with Sharplight (decoder, encoder, hybrid and multimodal families) with
/// <see cref="ModelAdapterRegistry"/>. <see cref="LlmModel"/> calls this itself; call it directly only when building
/// <see cref="ModelDefinition"/>s from an importer without going through <see cref="LlmModel"/>.
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
        ModelAdapterRegistry.Register(new NemotronHAdapter());
        ModelAdapterRegistry.Register(new HybridAdapter());
        ModelAdapterRegistry.Register(new Rwkv7Adapter());
        ModelAdapterRegistry.Register(new Rwkv6Adapter());
    }
}
