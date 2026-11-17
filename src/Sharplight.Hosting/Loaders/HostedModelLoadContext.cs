using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Hosting.Loaders;

/// <summary>What a <see cref="IHostedModelLoaderFactory"/> gets: the model, the host's options, the placement policy and the services.</summary>
/// <param name="Settings">The model's effective settings.</param>
/// <param name="Options">The hosting options (hooks such as <see cref="ModelHostingOptions.ConfigureLoad"/>).</param>
/// <param name="Placement">Chooses the devices; call it from <see cref="IHostedModelLoader.Prepare"/> so each load sees the current memory.</param>
/// <param name="Services">The application's services (model loader, task client provider, ...).</param>
public sealed record HostedModelLoadContext(HostedModelSettings Settings, ModelHostingOptions Options, IModelPlacementPolicy Placement, IServiceProvider Services);
