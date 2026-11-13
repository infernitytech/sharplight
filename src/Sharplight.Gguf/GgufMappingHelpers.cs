namespace Sharplight.Gguf;

/// <summary>Hyper-parameter conventions shared by several mappings.</summary>
internal static class GgufMappingHelpers
{
    /// <summary>Default RoPE base frequency when the file does not store one.</summary>
    public const float DefaultRopeBase = 10000f;

    /// <summary>Adds <c>{prefix}context_length</c> when the file states a training context.</summary>
    public static void AddContextLength(Dictionary<string, object> md, string prefix, GgufMetadataView g)
    {
        if (g.Has("context_length"))
        {
            md[prefix + "context_length"] = g.Int("context_length");
        }
    }

    /// <summary>
    /// Layer flags (1 = sliding window) for the llama.cpp convention "every <paramref name="pattern"/>-th layer is global":
    /// layer <c>il</c> slides when <c>il % pattern &lt; pattern - 1</c>.
    /// </summary>
    public static int[] SlidingLayers(int layerCount, int pattern)
        => Enumerable.Range(0, layerCount).Select(il => il % pattern < pattern - 1 ? 1 : 0).ToArray();
}
