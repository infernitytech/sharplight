using Sharplight.Models;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>A persistent state of the model (KV cache, recurrent state).</summary>
/// <param name="Name">State name.</param>
/// <param name="Type">Element type.</param>
/// <param name="Shape">Shape (symbolic dimensions included).</param>
/// <param name="Kind">State kind.</param>
/// <param name="Group">Placement group (layer).</param>
internal sealed record StateRow(string Name, string Type, string Shape, string Kind, string Group)
{
    /// <summary>Type and shape, e.g. <c>f16[ctx, 8, 128]</c>.</summary>
    public string TypeShape => Type + Shape;

    /// <summary>The row of a state definition.</summary>
    public static StateRow From(StateDefinition state) =>
        new(state.Name, state.Type.Name, state.Shape.ToString(), state.Kind.ToString(), state.Group);
}
