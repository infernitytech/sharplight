using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>What to load and how.</summary>
/// <param name="Id">The id the model is known by (in the chat and to API clients); unique among resident models.</param>
/// <param name="Path">Package, GGUF file or checkpoint directory.</param>
/// <param name="Settings">Device, context, batching, KV cache and speculation.</param>
/// <param name="DisabledFeatures">Device features not to use.</param>
/// <param name="Projector">Vision projector file (GGUF <c>mmproj</c>) to load with the model.</param>
/// <param name="Capabilities">What the library knows about the model before it is loaded.</param>
/// <param name="Kind">What the model does (<see cref="Sharplight.Models.ModelKinds"/>); the runtime loads it through the matching task client.</param>
/// <param name="Auxiliary">Further files by role that belong to the model (for example the <c>vocoder</c> of a speech synthesis model).</param>
internal sealed record ModelLoadRequest(string Id, string Path, LoadSettings Settings, IReadOnlyList<string> DisabledFeatures, string? Projector = null,
    ModelCapabilities Capabilities = ModelCapabilities.None, string Kind = Sharplight.Models.ModelKinds.TextGeneration,
    IReadOnlyList<(string Role, string Path)>? Auxiliary = null);
