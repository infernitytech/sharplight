using Microsoft.Extensions.Options;
using Sharplight.Distributed;
using Sharplight.Distributed.Transport;
using Sharplight.Hosting.Fit;
using Sharplight.Runtime;

namespace Sharplight.Hosting;

/// <summary>
/// The default placement: named devices are used as given (several split the model in proportion to their free memory beyond the
/// reserve, as <see cref="FitPlanner"/> assumes); <c>auto</c> takes the GPU with the most free memory beyond the reserve, preferring one
/// where the estimate fits, and the CPU when there is no GPU. With CPU offload a GPU placement gets the CPU as fallback.
/// </summary>
/// <param name="devices">The devices.</param>
/// <param name="options">The hosting options (the reserve).</param>
/// <param name="remoteDevices">The segment workers a model's <see cref="HostedModelSettings.Workers"/> name: they follow the local devices in
/// pipeline order, each weighted by its free memory beyond the reserve.</param>
public sealed class DefaultModelPlacementPolicy(DeviceInventory devices, IOptions<ModelHostingOptions> options, RemoteDevicePool remoteDevices) : IModelPlacementPolicy
{
    /// <inheritdoc/>
    public DevicePlacement Place(HostedModelSettings model, long estimatedBytes)
    {
        ArgumentNullException.ThrowIfNull(model);
        long reserve = options.Value.Reserve;
        var chosen = devices.Resolve(model.Device).Select(device => device ?? ChooseAuto(estimatedBytes, reserve)).Distinct().ToList();
        var opened = chosen.Select(devices.Open).Concat(model.Workers.Select(w => remoteDevices.Open(SegmentEndpoint.Parse(w)))).ToList();
        IReadOnlyList<DeviceSlot> slots = opened.Count == 1
            ? [new DeviceSlot { Device = opened[0] }]
            : [.. opened.Select(device => new DeviceSlot { Device = device, Weight = FitPlanner.SplitWeight(device.GetMemoryInfo().Available, reserve) })];
        bool offload = model.CpuOffload && chosen[0].IsGpu && devices.Cpu is not null;
        var fallback = offload ? new DeviceSlot { Device = devices.Open(devices.Cpu!) } : null;
        return new DevicePlacement(slots, fallback);
    }

    private HostDevice ChooseAuto(long estimatedBytes, long reserve)
    {
        var gpus = devices.Devices.Where(device => device.IsGpu).ToList();
        if (gpus.Count == 0)
        {
            return devices.Cpu ?? throw new InvalidOperationException("No device is available.");
        }

        var ranked = gpus
            .Select(device => (Device: device, Available: devices.Open(device).GetMemoryInfo().Available - reserve))
            .OrderByDescending(entry => entry.Available >= estimatedBytes)
            .ThenByDescending(entry => entry.Available)
            .ToList();
        return ranked[0].Device;
    }
}
