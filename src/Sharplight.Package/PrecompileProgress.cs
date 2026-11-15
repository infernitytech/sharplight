using Sharplight.Runtime;

namespace Sharplight.Package;

/// <summary>Progress of <see cref="PackagePrecompiler"/> for one provider and device.</summary>
/// <param name="Provider">The execution provider; empty for the final writing step.</param>
/// <param name="Device">The device (display name); empty for provider-level and writing steps.</param>
/// <param name="Stage">What is happening.</param>
/// <param name="Message">A short description (e.g. why a provider was skipped).</param>
/// <param name="Kernels">Kernel compilation progress during <see cref="PrecompileStage.Portable"/>, else <see langword="null"/>.</param>
public sealed record PrecompileProgress(string Provider, string Device, PrecompileStage Stage, string Message, LoadProgress? Kernels = null);
