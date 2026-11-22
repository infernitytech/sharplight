using Sharplight.Runtime.Experts;

namespace Sharplight.Cpu;

/// <summary>
/// The routed-expert weights of an expert product that stay on disk (<see cref="Runtime.CompileOptions.DiskExperts"/>): the op holds the
/// experts a pass selects in the store's cache while it computes them.
/// </summary>
/// <param name="Store">The store.</param>
/// <param name="First">The op's (first) weight.</param>
/// <param name="Second">The second weight of a gate / up pair, or null.</param>
internal sealed record DiskExpertBinding(ExpertStore Store, DiskExpertTensor First, DiskExpertTensor? Second);
