namespace Sharplight.Runtime;

/// <summary>An execution provider and whether it can run on this machine (<see cref="ExecutionProviderCatalog.Discover"/>).</summary>
/// <param name="Provider">The provider; owned by the catalog, do not dispose it.</param>
/// <param name="Availability">The result of <see cref="IExecutionProvider.CheckAvailability"/>.</param>
public sealed record DiscoveredProvider(IExecutionProvider Provider, ProviderAvailability Availability);
