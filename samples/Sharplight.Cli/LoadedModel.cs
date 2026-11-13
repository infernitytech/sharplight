using Sharplight.Llm;
using Sharplight.Runtime;

namespace Sharplight.Cli;

/// <summary>A loaded model together with its decode session and the devices it runs on; disposing releases all of them in dependency order.</summary>
internal sealed class LoadedModel : IDisposable
{
    private readonly IReadOnlyList<DeviceSlot> _devices;
    private readonly DeviceSlot? _fallback;

    public LoadedModel(LlmModel model, InferenceSession session, IReadOnlyList<DeviceSlot> devices, DeviceSlot? fallback)
    {
        Model = model;
        Session = session;
        _devices = devices;
        _fallback = fallback;
    }

    public LlmModel Model { get; }

    public InferenceSession Session { get; }

    public void Dispose()
    {
        Session.Dispose();
        Model.Dispose();
        foreach (var slot in _devices)
        {
            slot.Device.Dispose();
        }

        _fallback?.Device.Dispose();
    }
}
