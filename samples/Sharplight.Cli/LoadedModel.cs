using Sharplight.Llm;
using Sharplight.Runtime;

namespace Sharplight.Cli;

/// <summary>
/// A loaded model together with its decode session (compiled on first use) and the devices it runs on; disposing releases all of
/// them in dependency order.
/// </summary>
internal sealed class LoadedModel : IDisposable
{
    private readonly IReadOnlyList<DeviceSlot> _devices;
    private readonly DeviceSlot? _fallback;
    private readonly string _graph;
    private InferenceSession? _session;

    public LoadedModel(LlmModel model, string graph, IReadOnlyList<DeviceSlot> devices, DeviceSlot? fallback)
    {
        Model = model;
        _graph = graph;
        _devices = devices;
        _fallback = fallback;
    }

    public LlmModel Model { get; }

    /// <summary>The single-sequence session; speculative generation uses a batch decoder instead and never compiles it.</summary>
    public InferenceSession Session => _session ??= Model.CreateSession(_graph);

    public void Dispose()
    {
        _session?.Dispose();
        Model.Dispose();
        foreach (var slot in _devices)
        {
            slot.Device.Dispose();
        }

        _fallback?.Device.Dispose();
    }
}
