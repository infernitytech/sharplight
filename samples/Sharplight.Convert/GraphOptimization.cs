using Sharplight.Models;

namespace Sharplight.Convert;

/// <summary>
/// Hook for the device-independent graph optimization phase (docs/model-format.md, "Graph optimization phase") that
/// <c>convert</c> runs before saving.
/// </summary>
internal static class GraphOptimization
{
    /// <summary>
    /// Returns the definition to store. Placeholder: the optimizer (<c>GraphOptimizer</c> in Sharplight.Core) is being built
    /// separately; until it lands the definition is stored as built and this reports that nothing ran.
    /// </summary>
    /// <param name="definition">The built model.</param>
    /// <param name="applied">Whether an optimization pipeline ran.</param>
    public static ModelDefinition Run(ModelDefinition definition, out bool applied)
    {
        // TODO: call the Sharplight.Core graph optimizer here once it exists.
        applied = false;
        return definition;
    }
}
