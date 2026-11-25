using Sharplight.Hosting.Fit;
using Sharplight.Llm.Serving.Hosting;
using Sharplight.Runtime;

namespace Sharplight.Hosting;

/// <summary>
/// Owns the <see cref="ModelHost"/> of an application and manages its models: registration from <see cref="HostedModelOptions"/>, load,
/// unload and removal, fit planning against the live device memory, and a status snapshot (free memory per device, loaded models, queue
/// depth, throughput) for health checks, dashboards and external cluster coordinators. Thread-safe.
/// </summary>
public interface IModelHostManager
{
    /// <summary>The model host: acquire leases on models from it (<see cref="ModelHost.AcquireAsync"/>).</summary>
    public ModelHost Host { get; }

    /// <summary>The devices models are placed on.</summary>
    public DeviceInventory Devices { get; }

    /// <summary>Registers a model; it loads on its first request, or now when it is not lazy and startup loading already happened.</summary>
    /// <param name="model">The model.</param>
    /// <returns>The effective settings it was registered with.</returns>
    /// <exception cref="ArgumentException">The model has no name or path, or no loader factory accepts it.</exception>
    /// <exception cref="InvalidOperationException">The name is already registered.</exception>
    public HostedModelSettings Register(HostedModelOptions model);

    /// <summary>The effective settings of a registered model, or null.</summary>
    /// <param name="name">The registered name.</param>
    public HostedModelSettings? Find(string name);

    /// <summary>Loads a model now (no-op when it is loaded).</summary>
    /// <param name="name">The registered name.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    /// <returns>A task that completes when the model is loaded.</returns>
    /// <exception cref="ModelNotFoundException">No model has this name.</exception>
    /// <exception cref="ModelDoesNotFitException">It cannot be made to fit.</exception>
    public Task LoadAsync(string name, CancellationToken cancellationToken = default);

    /// <summary>Unloads a model (it stays registered and reloads on demand). See <see cref="ModelHost.UnloadAsync"/>.</summary>
    /// <param name="name">The registered name.</param>
    /// <param name="drainTimeout">How long running requests may take, or null for the host's default.</param>
    /// <param name="force">Cut running requests off after the timeout instead of throwing.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    /// <returns>False when it was not loaded.</returns>
    public Task<bool> UnloadAsync(string name, TimeSpan? drainTimeout = null, bool force = false, CancellationToken cancellationToken = default);

    /// <summary>Unloads a model and removes its registration.</summary>
    /// <param name="name">The registered name.</param>
    /// <param name="drainTimeout">How long running requests may take, or null for the host's default.</param>
    /// <param name="force">Cut running requests off after the timeout instead of throwing.</param>
    /// <param name="cancellationToken">Stops waiting.</param>
    /// <returns>False when it was not registered.</returns>
    public Task<bool> RemoveAsync(string name, TimeSpan? drainTimeout = null, bool force = false, CancellationToken cancellationToken = default);

    /// <summary>A snapshot of the devices, models and load.</summary>
    /// <returns>The status.</returns>
    public HostStatus GetStatus();

    /// <summary>
    /// Whether a model of <paramref name="footprint"/> would load on the devices <paramref name="device"/> names, against their live free
    /// memory, the reserve and the resident models (which the host may evict), as the host admits a load.
    /// </summary>
    /// <param name="footprint">The estimated footprint (e.g. <c>LlmModel.EstimateFootprint</c>).</param>
    /// <param name="device">The device specification (<see cref="DeviceSpecifications"/>); null = the default.</param>
    /// <param name="cpuOffload">Layers that do not fit may run on the CPU.</param>
    /// <param name="expertsOnDisk">Routed experts that do not fit host memory may stay on disk (with <paramref name="cpuOffload"/>).</param>
    /// <returns>The verdict and the figures behind it.</returns>
    public FitReport PlanFit(ModelFootprint footprint, string? device = null, bool cpuOffload = false, bool expertsOnDisk = false);
}
