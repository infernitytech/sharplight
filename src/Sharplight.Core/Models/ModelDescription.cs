using System.Diagnostics.CodeAnalysis;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;

namespace Sharplight.Models;

/// <summary>
/// The neutral description of what a model is and which parts it consists of: its kind (<see cref="ModelKinds"/>, open-ended), its
/// components (<see cref="ModelComponent"/>: graphs, weights, configuration, attachments) and model-level configuration. One
/// description serves every model kind; architecture adapters set it on the <see cref="ModelDefinition"/> they build
/// (<see cref="ModelDefinition.Description"/>), the <c>.slm</c> package stores it as a JSON section (docs/model-format.md), and the
/// task layer picks the client implementation by its kind.
/// <para>
/// Models without a description (every LLM package written before it existed) resolve to a text-generation model with one
/// <see cref="ModelComponentRoles.Decoder"/> component that runs all graphs (<see cref="Resolve"/>).
/// </para>
/// </summary>
public sealed class ModelDescription
{
    /// <summary>
    /// The schema version written by this library. Additions (new optional fields) keep the version, readers ignore fields they do not
    /// know; only an incompatible change raises it, and <see cref="FromJson(string)"/> rejects versions newer than this one.
    /// </summary>
    public const int CurrentSchemaVersion = 1;

    private static readonly JsonSerializerOptions s_writeOptions = new(ModelDescriptionJsonContext.Default.Options)
    {
        Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
    };

    /// <summary>Schema version of the serialized form (<see cref="CurrentSchemaVersion"/> for descriptions created here).</summary>
    public int SchemaVersion { get; init; } = CurrentSchemaVersion;

    /// <summary>Model kind (<see cref="ModelKinds"/>; open-ended, lowercase and hyphenated).</summary>
    public required string Kind { get; init; }

    /// <summary>The components in declaration order; roles are unique.</summary>
    public IReadOnlyList<ModelComponent> Components { get; init; } = [];

    /// <summary>Model-level configuration as JSON values (e.g. the input sample rate of a speech model).</summary>
    public IReadOnlyDictionary<string, JsonElement> Config { get; init; } = new Dictionary<string, JsonElement>();

    /// <summary>Gets a component by role.</summary>
    /// <param name="role">The component role.</param>
    /// <exception cref="KeyNotFoundException">The model has no component with this role.</exception>
    public ModelComponent Component(string role)
        => TryGetComponent(role, out var component)
            ? component
            : throw new KeyNotFoundException($"The {Kind} model has no '{role}' component. Components: {string.Join(", ", Components.Select(c => c.Role))}");

    /// <summary>Gets a component by role.</summary>
    /// <param name="role">The component role.</param>
    /// <param name="component">The component, or <see langword="null"/>.</param>
    /// <returns>False when the model has no component with this role.</returns>
    public bool TryGetComponent(string role, [NotNullWhen(true)] out ModelComponent? component)
    {
        component = Components.FirstOrDefault(c => c.Role == role);
        return component is not null;
    }

    /// <summary>Reads a model-level configuration value, or returns <paramref name="defaultValue"/> when it is absent.</summary>
    /// <typeparam name="T">string, bool or a number type.</typeparam>
    /// <param name="key">The configuration key.</param>
    /// <param name="defaultValue">The value for an absent key.</param>
    /// <exception cref="FormatException">The value has another JSON type.</exception>
    public T ConfigValue<T>(string key, T defaultValue) => Config.TryGetValue(key, out var value) ? JsonValues.Convert<T>(value) : defaultValue;

    /// <summary>
    /// Checks the description against itself and, when given, against the definition it describes: the kind and every role are
    /// non-empty, roles are unique, and every graph and attachment a component names exists.
    /// </summary>
    /// <param name="definition">The described model, or <see langword="null"/> to check only the description itself.</param>
    /// <exception cref="InvalidDataException">The description is inconsistent.</exception>
    public void Validate(ModelDefinition? definition = null)
    {
        if (string.IsNullOrWhiteSpace(Kind))
        {
            throw new InvalidDataException("A model description needs a kind.");
        }

        var roles = new HashSet<string>(StringComparer.Ordinal);
        foreach (var component in Components)
        {
            if (string.IsNullOrWhiteSpace(component.Role) || !roles.Add(component.Role))
            {
                throw new InvalidDataException($"Model description: empty or duplicate component role '{component.Role}'.");
            }

            if (definition is null)
            {
                continue;
            }

            foreach (var graph in component.Graphs.Where(g => !definition.Graphs.ContainsKey(g)))
            {
                throw new InvalidDataException($"Component '{component.Role}' names the graph '{graph}', which the model does not have.");
            }

            foreach (var attachment in component.Attachments.Values.Where(a => !definition.Attachments.ContainsKey(a)))
            {
                throw new InvalidDataException($"Component '{component.Role}' names the attachment '{attachment}', which the model does not have.");
            }
        }
    }

    /// <summary>
    /// The description of <paramref name="definition"/>: its own <see cref="ModelDefinition.Description"/>, or, for models without one,
    /// a text-generation model with one <see cref="ModelComponentRoles.Decoder"/> component that runs all graphs.
    /// </summary>
    /// <param name="definition">The model.</param>
    public static ModelDescription Resolve(ModelDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        return definition.Description ?? new ModelDescription
        {
            Kind = ModelKinds.TextGeneration,
            Components = [new ModelComponent { Role = ModelComponentRoles.Decoder, Graphs = [.. definition.Graphs.Keys.Order(StringComparer.Ordinal)] }],
        };
    }

    /// <summary>Serializes the description as indented UTF-8 JSON (camelCase property names).</summary>
    public string ToJson() => JsonSerializer.Serialize(this, (JsonTypeInfo<ModelDescription>)s_writeOptions.GetTypeInfo(typeof(ModelDescription)));

    /// <summary>Reads a description written by <see cref="ToJson"/>.</summary>
    /// <param name="json">The JSON text.</param>
    /// <exception cref="InvalidDataException">The text is not a valid description.</exception>
    /// <exception cref="NotSupportedException">The description has a newer, incompatible schema version.</exception>
    public static ModelDescription FromJson(string json)
    {
        ModelDescription? description;
        try
        {
            description = JsonSerializer.Deserialize(json, ModelDescriptionJsonContext.Default.ModelDescription);
        }
        catch (JsonException error)
        {
            throw new InvalidDataException($"Invalid model description: {error.Message}", error);
        }

        if (description is null)
        {
            throw new InvalidDataException("Invalid model description: null.");
        }

        if (description.SchemaVersion > CurrentSchemaVersion)
        {
            throw new NotSupportedException(
                $"Model description schema version {description.SchemaVersion} is newer than the supported version {CurrentSchemaVersion}.");
        }

        description.Validate();
        return description;
    }
}
