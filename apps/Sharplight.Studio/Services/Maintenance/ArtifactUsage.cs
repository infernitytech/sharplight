namespace Sharplight.Studio.Services.Maintenance;

/// <summary>Precompiled artifacts of one provider and device stored in a package.</summary>
/// <param name="Provider">The execution provider (<c>vulkan</c>, <c>cuda</c>).</param>
/// <param name="Device">The device or device class the code is for, or empty for portable code.</param>
/// <param name="Description">The artifact sets, e.g. <c>spirv portable, pipeline-cache</c>.</param>
/// <param name="Bytes">Bytes of the sets.</param>
/// <param name="Entries">Number of kernels or entries.</param>
internal sealed record ArtifactUsage(string Provider, string Device, string Description, long Bytes, int Entries);
