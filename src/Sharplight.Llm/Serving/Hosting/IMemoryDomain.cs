using Sharplight.Runtime;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A pool of memory models are charged against (normally one GPU). Same <see cref="Id"/> = same pool.</summary>
public interface IMemoryDomain
{
    public string Id { get; }
    public string Name { get; }
    public DeviceMemoryInfo GetMemoryInfo();
}
