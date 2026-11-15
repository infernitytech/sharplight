using Sharplight.Studio.Models;

namespace Sharplight.Studio.Services.Library;

/// <summary>What <see cref="ModelInspector"/> reads from a model file without loading it.</summary>
/// <param name="Capabilities">What the model can do (from its graphs and chat template).</param>
/// <param name="ContextLength">The context length it was trained for, when known.</param>
/// <param name="Architecture">The architecture id, when known.</param>
/// <param name="Kind">The model kind (<see cref="Sharplight.Models.ModelKinds"/>).</param>
/// <param name="KindInferred">The kind is a guess from the file type: the file has no model description to read.</param>
/// <param name="Precompiled">Descriptors (formatted) of the precompiled artifact sets stored in the package.</param>
internal sealed record InspectedModel(ModelCapabilities Capabilities, int? ContextLength, string? Architecture, string Kind, bool KindInferred,
    IReadOnlyList<string> Precompiled);
