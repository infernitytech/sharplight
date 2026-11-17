using Sharplight.Runtime;

namespace Sharplight.Hosting;

/// <summary>Where one load of a model goes: the devices in pipeline order and the optional CPU fallback for offloaded layers.</summary>
/// <param name="Devices">The devices (opened); the model is split over them by their <see cref="DeviceSlot.Weight"/>.</param>
/// <param name="Fallback">Device for layers that do not fit and unsupported ops, or null.</param>
public sealed record DevicePlacement(IReadOnlyList<DeviceSlot> Devices, DeviceSlot? Fallback);
