using Sharplight.Runtime;

namespace Sharplight.Studio.Services.Fit;

/// <summary>The longest contexts that fit a model with its other load settings, from the engine's footprint estimates.</summary>
/// <param name="MaxContext">The longest context tried (what the model was trained for, else <see cref="ContextSizer.DefaultMaxContext"/>).</param>
/// <param name="OnDevice">The longest context that fits on the chosen device without offloading (counting automatic unloads); null when none does.</param>
/// <param name="OnDeviceFootprint">Its footprint.</param>
/// <param name="WithOffload">
/// The longest context that fits with CPU offload, when that is longer than <paramref name="OnDevice"/> (GPU loads only); null otherwise.
/// </param>
/// <param name="WithOffloadFootprint">Its footprint.</param>
/// <param name="ExpertOffload">The offload at <paramref name="WithOffload"/> moves only routed experts (a mixture-of-experts model).</param>
internal sealed record ContextFit(int MaxContext, int? OnDevice, ModelFootprint? OnDeviceFootprint, int? WithOffload = null,
    ModelFootprint? WithOffloadFootprint = null, bool ExpertOffload = false);
