using Sharplight.Package.Import;
using Sharplight.Runtime;

namespace Sharplight.Hub;

/// <summary>The device a pulled package is precompiled for, and how.</summary>
/// <param name="Provider">The device's execution provider.</param>
/// <param name="Device">The device.</param>
/// <param name="Options">Context length, batch size and tiers; use the values the model will run with.</param>
public sealed record PrecompileTarget(IExecutionProvider Provider, DeviceInfo Device, PrecompileOptions Options);
