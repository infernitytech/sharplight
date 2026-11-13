using Sharplight.Architectures;
using Sharplight.Models;

namespace Sharplight.Llm;

/// <summary>Registers the architecture adapters shipped with Sharplight.</summary>
internal static class BuiltinAdapters
{
    private static int s_done;
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
    }
}
