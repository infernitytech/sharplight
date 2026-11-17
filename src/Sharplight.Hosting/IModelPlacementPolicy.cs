namespace Sharplight.Hosting;

/// <summary>
/// Decides the devices of a model each time it loads (so <c>auto</c> placement follows the memory that is free at that moment).
/// Replace the registered policy to implement other placement rules; the default is <see cref="DefaultModelPlacementPolicy"/>.
/// </summary>
public interface IModelPlacementPolicy
{
    /// <summary>The devices for loading <paramref name="model"/>.</summary>
    /// <param name="model">The model's settings (<see cref="HostedModelSettings.Device"/> holds what was asked for).</param>
    /// <param name="estimatedBytes">A rough footprint estimate (from the file sizes), for choosing among devices.</param>
    /// <returns>The placement.</returns>
    /// <exception cref="InvalidOperationException">The specification names no available device.</exception>
    public DevicePlacement Place(HostedModelSettings model, long estimatedBytes);
}
