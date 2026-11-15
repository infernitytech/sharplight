using Sharplight.Models;

namespace Sharplight.Tasks;

/// <summary>Describes what a task client (<see cref="ITaskClient"/>) runs.</summary>
/// <param name="Kind">The model kind (<see cref="ModelKinds"/>).</param>
/// <param name="ModelName">The model's display name (<see cref="ModelDefinition.Name"/>); empty when unknown.</param>
/// <param name="Architecture">The canonical architecture id (<see cref="ModelDefinition.Architecture"/>).</param>
public sealed record TaskClientMetadata(string Kind, string ModelName, string Architecture)
{
    /// <summary>The metadata of a client for <paramref name="definition"/>, with the kind of its (resolved) description.</summary>
    /// <param name="definition">The model the client runs.</param>
    public static TaskClientMetadata For(ModelDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        return new TaskClientMetadata(ModelDescription.Resolve(definition).Kind, definition.Name, definition.Architecture);
    }
}
