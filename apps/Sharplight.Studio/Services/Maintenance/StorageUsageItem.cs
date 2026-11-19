namespace Sharplight.Studio.Services.Maintenance;

/// <summary>One slice of a storage location's usage.</summary>
/// <param name="Key">Stable key (<see cref="UsageKeys"/>).</param>
/// <param name="Title">What it is, e.g. <c>Precompiled kernels</c>.</param>
/// <param name="Bytes">Bytes used.</param>
internal sealed record StorageUsageItem(string Key, string Title, long Bytes);
