using Sharplight.Graph;
using Sharplight.Runtime;

namespace Sharplight.Architectures;

/// <summary>
/// The left contexts of the causal convolutions of one compiled <see cref="VibeVoiceTokenizerGraph"/> stream: arrays the host feeds to the graph's cache inputs and refreshes from its cache
/// outputs after every run, so consecutive runs continue one signal. Zero at the start (or after <see cref="Reset"/>), as the convolutions' left padding.
/// </summary>
internal sealed class VibeVoiceConvCaches
{
    private readonly CompiledModel _model;
    private readonly List<(Value Input, string Output, float[] Data)> _caches = [];

    /// <summary>Collects the cache inputs of <paramref name="model"/> whose names start with <paramref name="prefix"/> followed by <see cref="VibeVoiceTokenizerGraph.CacheInfix"/>.</summary>
    /// <param name="model">The compiled graph.</param>
    /// <param name="prefix">The prefix of the tokenizer's caches in the graph (for example <c>acoustic.</c>).</param>
    public VibeVoiceConvCaches(CompiledModel model, string prefix)
    {
        _model = model;
        string start = prefix + VibeVoiceTokenizerGraph.CacheInfix;
        foreach (var input in model.Graph.Inputs.Where(i => i.Input!.Name.StartsWith(start, StringComparison.Ordinal)))
        {
            string name = input.Input!.Name;
            _caches.Add((input, prefix + VibeVoiceTokenizerGraph.CacheOutInfix + name[start.Length..], new float[input.Shape.Elements()]));
        }
    }

    /// <summary>Clears every context.</summary>
    public void Reset()
    {
        foreach (var (_, _, data) in _caches)
        {
            Array.Clear(data);
        }
    }

    /// <summary>Feeds the contexts to the graph's cache inputs.</summary>
    public void Bind()
    {
        foreach (var (input, _, data) in _caches)
        {
            _model.SetInput<float>(input, data);
        }
    }

    /// <summary>Reads the contexts for the next run from the cache outputs of the run that just finished.</summary>
    public void Collect()
    {
        foreach (var (_, output, data) in _caches)
        {
            _model.GetOutput<float>(output, data);
        }
    }

    /// <summary>The number of cached contexts.</summary>
    public int Count => _caches.Count;
}
