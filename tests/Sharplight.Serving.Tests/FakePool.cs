using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;

namespace Sharplight.Serving.Tests;

/// <summary>A memory pool with a fixed capacity; models charge it on load and give it back on dispose. Other applications' use is simulated with <see cref="External"/>.</summary>
public sealed class FakePool(string name, long capacity) : IMemoryDomain
{
    private long _ours, _external;
    public string Id => name;
    public string Name => name;
    public long External { get => Interlocked.Read(ref _external); set => Interlocked.Exchange(ref _external, value); }
    public long Ours => Interlocked.Read(ref _ours);
    public void Charge(long b) => Interlocked.Add(ref _ours, b);
    public DeviceMemoryInfo GetMemoryInfo() => new(capacity, capacity, Ours + External, Ours, true);
}
