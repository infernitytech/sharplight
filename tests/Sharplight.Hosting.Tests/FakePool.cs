using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;

namespace Sharplight.Hosting.Tests;

/// <summary>A memory pool with a fixed capacity; models charge it on load and give it back on dispose.</summary>
public sealed class FakePool(string id, long capacity) : IMemoryDomain
{
    private long _ours;

    public string Id => id;

    public string Name => id;

    public void Charge(long bytes) => Interlocked.Add(ref _ours, bytes);

    public DeviceMemoryInfo GetMemoryInfo() => new(capacity, capacity, Interlocked.Read(ref _ours), Interlocked.Read(ref _ours), true);
}
