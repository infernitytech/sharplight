using Sharplight.Runtime;

namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A pool of memory models are charged against (normally one GPU). Same <see cref="Id"/> = same pool.</summary>
public interface IMemoryDomain
{
    /// <summary>Stable identifier; domains with the same id share one pool.</summary>
    public string Id { get; }
    /// <summary>Display name (for example the GPU name).</summary>
    public string Name { get; }
    /// <summary>Current total and free bytes of the pool.</summary>
    public DeviceMemoryInfo GetMemoryInfo();
}
