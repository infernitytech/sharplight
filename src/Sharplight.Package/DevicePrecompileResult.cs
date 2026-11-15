namespace Sharplight.Package;

/// <summary>What <see cref="PackagePrecompiler"/> produced for one device.</summary>
/// <param name="Provider">The execution provider.</param>
/// <param name="Device">The device (display name).</param>
/// <param name="Portable">Stage one (portable code), or <see langword="null"/> when the provider has none for this device.</param>
/// <param name="DeviceCode">Stage two (device code), or <see langword="null"/> when not produced (no device stage, or not requested).</param>
/// <param name="Note">Why a stage produced nothing (e.g. "nothing to precompile"), or <see langword="null"/>.</param>
public sealed record DevicePrecompileResult(string Provider, string Device, CodeSetSummary? Portable, CodeSetSummary? DeviceCode, string? Note);
