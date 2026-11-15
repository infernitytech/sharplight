using Sharplight.Runtime;

namespace Sharplight.Package.Tests;

/// <summary>
/// An execution provider with one <see cref="RecordingArtifactDevice"/> per given driver version (equal drivers: identical GPUs) that
/// implements both precompilation stages: the portable stage runs the workload on a recording device of the profile, the device stage
/// returns one <c>fake/binary/...</c> entry per kernel. Counts the calls of each stage.
/// </summary>
/// <param name="cpu">The CPU device the recording devices compile on.</param>
/// <param name="drivers">Driver version of each device.</param>
public sealed class FakeCodeProvider(IComputeDevice cpu, IReadOnlyList<string> drivers) : IExecutionProvider
{
    public string Name => "fake";

    public string? PortableCodeFormat => "spirv_1_6";

    public int PortableCalls { get; private set; }

    public int DeviceCalls { get; private set; }

    /// <summary>Fails the device stage of devices with this driver (a driver error), or null.</summary>
    public string? FailingDriver { get; init; }

    public ProviderAvailability CheckAvailability() => ProviderAvailability.Available();

    public IReadOnlyList<DeviceInfo> GetDevices() => [.. drivers.Select((driver, index) => new RecordingArtifactDevice(cpu, driver, index).Info)];

    public IComputeDevice OpenDevice(DeviceInfo device, DeviceOptions? options = null) => new RecordingArtifactDevice(cpu, device.Driver, device.Index);

    public bool CanCompileFor(DeviceProfile profile) => profile.Provider == Name;

    public Task<PortableCodeSet> CompilePortableAsync(PortableCompilationRequest request, CancellationToken cancellationToken = default)
    {
        PortableCalls++;
        var device = new RecordingArtifactDevice(cpu);
        var artifacts = new CompilationArtifacts(request.Stored);
        request.Workload(device, artifacts, cancellationToken);
        return Task.FromResult(artifacts.PortableCodeFor(device));
    }

    public Task<DeviceCodeSet> CompileDeviceSpecificAsync(IComputeDevice device, PortableCodeSet portable, CancellationToken cancellationToken = default)
    {
        DeviceCalls++;
        if (device.Info.Driver == FailingDriver)
        {
            throw new InvalidOperationException("driver error");
        }

        var target = device.ArtifactTargets.Single(t => t.Tier == ArtifactTiers.Binary);
        var entries = portable.Entries.ToDictionary(e => "fake/binary/" + e.Key, e => (ReadOnlyMemory<byte>)new byte[] { 2, 2 });
        return Task.FromResult(new DeviceCodeSet(Name, device.Info.Name, target.Descriptor, entries));
    }

    public void Dispose()
    {
    }
}
