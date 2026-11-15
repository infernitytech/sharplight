using System.Globalization;

namespace Sharplight.Tasks;

/// <summary>
/// The neutral description of what a model is for and which parts it consists of: the task id (<see cref="ModelTasks"/>), the
/// components (graphs and component-scoped metadata) and task settings (scheduler configuration, audio feature parameters, sample
/// rates, ...). Architecture adapters write it into <see cref="Models.ModelDefinition.Properties"/> (<see cref="WriteTo"/>) and task
/// loaders read it back (<see cref="Read"/>), so a model saved as an <c>.slm</c> package, where no adapter runs, still selects the
/// right task utility and finds its parts. The property layout is in <see cref="TaskPropertyKeys"/> and docs/model-format.md.
/// Models without a descriptor (decoder LLMs, embedding models and ONNX graphs saved before it existed) keep loading through their
/// task-specific loaders.
/// </summary>
public sealed class ModelTaskDescriptor
{
    /// <summary>
    /// The layout version written by <see cref="WriteTo"/>. Additions (new settings, roles, optional keys) keep the version;
    /// only an incompatible change of the existing keys raises it, and <see cref="Read"/> rejects versions newer than this.
    /// </summary>
    public const int CurrentVersion = 1;

    /// <summary>Task id (<see cref="ModelTasks"/>; open-ended).</summary>
    public required string Kind { get; init; }

    /// <summary>The components in declaration order; role names are unique.</summary>
    public IReadOnlyList<TaskComponent> Components { get; init; } = [];

    /// <summary>
    /// Task settings by name (without the <see cref="TaskPropertyKeys.SettingPrefix"/>), e.g. <c>scheduler.prediction_type</c> or
    /// <c>audio.sample_rate</c>. Values are property values: strings, booleans, numbers or arrays of them.
    /// </summary>
    public IReadOnlyDictionary<string, object> Settings { get; init; } = new Dictionary<string, object>();

    /// <summary>Gets a component by role.</summary>
    /// <exception cref="KeyNotFoundException">The model has no component with this role.</exception>
    public TaskComponent Component(string role)
        => TryGetComponent(role, out var component)
            ? component
            : throw new KeyNotFoundException($"The {Kind} model has no '{role}' component. Components: {string.Join(", ", Components.Select(c => c.Role))}");

    /// <summary>Gets a component by role.</summary>
    /// <returns>False when the model has no component with this role.</returns>
    public bool TryGetComponent(string role, [System.Diagnostics.CodeAnalysis.NotNullWhen(true)] out TaskComponent? component)
    {
        component = Components.FirstOrDefault(c => c.Role == role);
        return component is not null;
    }

    /// <summary>Reads a setting, converting it to <typeparamref name="T"/>, or returns <paramref name="defaultValue"/> when it is absent.</summary>
    public T Setting<T>(string name, T defaultValue)
        => Settings.TryGetValue(name, out var value) ? (T)Convert.ChangeType(value, typeof(T), CultureInfo.InvariantCulture) : defaultValue;

    /// <summary>Reads the descriptor from model properties.</summary>
    /// <param name="properties">The <see cref="Models.ModelDefinition.Properties"/> of a model.</param>
    /// <returns>The descriptor, or <see langword="null"/> when the model has none (no <see cref="TaskPropertyKeys.Kind"/>).</returns>
    /// <exception cref="NotSupportedException">The descriptor was written in a newer, incompatible layout version.</exception>
    /// <exception cref="InvalidDataException">A component listed in <see cref="TaskPropertyKeys.Components"/> is malformed.</exception>
    public static ModelTaskDescriptor? Read(IReadOnlyDictionary<string, object> properties)
    {
        if (!properties.TryGetValue(TaskPropertyKeys.Kind, out var kindValue) || kindValue is not string kind)
        {
            return null;
        }

        int version = properties.TryGetValue(TaskPropertyKeys.Version, out var v) ? Convert.ToInt32(v, CultureInfo.InvariantCulture) : CurrentVersion;
        if (version > CurrentVersion)
        {
            throw new NotSupportedException($"Task descriptor version {version} is newer than the supported version {CurrentVersion}.");
        }

        var roles = properties.TryGetValue(TaskPropertyKeys.Components, out var r) ? r as string[] ?? [] : [];
        var components = new List<TaskComponent>(roles.Length);
        foreach (var role in roles)
        {
            if (string.IsNullOrEmpty(role) || components.Any(c => c.Role == role))
            {
                throw new InvalidDataException($"Task descriptor: empty or duplicate component role '{role}'.");
            }

            components.Add(new TaskComponent(role, ReadString(properties, GraphKey(role)), ReadString(properties, MetadataKey(role))));
        }

        var settings = new Dictionary<string, object>(StringComparer.Ordinal);
        foreach (var (key, value) in properties)
        {
            if (key.StartsWith(TaskPropertyKeys.SettingPrefix, StringComparison.Ordinal))
            {
                settings[key[TaskPropertyKeys.SettingPrefix.Length..]] = value;
            }
        }

        return new ModelTaskDescriptor { Kind = kind, Components = components, Settings = settings };
    }

    /// <summary>
    /// Writes the descriptor into model properties (an adapter calls this while building the definition). Existing descriptor keys
    /// for the same roles and settings are overwritten.
    /// </summary>
    /// <param name="properties">The properties of the definition being built.</param>
    /// <exception cref="ArgumentException">Two components share a role, or a role is empty.</exception>
    public void WriteTo(IDictionary<string, object> properties)
    {
        var roles = Components.Select(c => c.Role).ToArray();
        if (roles.Any(string.IsNullOrEmpty) || roles.Distinct(StringComparer.Ordinal).Count() != roles.Length)
        {
            throw new ArgumentException("Task component roles must be non-empty and unique.");
        }

        properties[TaskPropertyKeys.Version] = CurrentVersion;
        properties[TaskPropertyKeys.Kind] = Kind;
        properties[TaskPropertyKeys.Components] = roles;
        foreach (var component in Components)
        {
            if (component.Graph is not null)
            {
                properties[GraphKey(component.Role)] = component.Graph;
            }

            if (component.MetadataPrefix is not null)
            {
                properties[MetadataKey(component.Role)] = component.MetadataPrefix;
            }
        }

        foreach (var (name, value) in Settings)
        {
            properties[TaskPropertyKeys.SettingPrefix + name] = value;
        }
    }

    private static string GraphKey(string role) => TaskPropertyKeys.ComponentPrefix + role + TaskPropertyKeys.ComponentGraphSuffix;

    private static string MetadataKey(string role) => TaskPropertyKeys.ComponentPrefix + role + TaskPropertyKeys.ComponentMetadataSuffix;

    private static string? ReadString(IReadOnlyDictionary<string, object> properties, string key)
        => properties.TryGetValue(key, out var value) ? value as string : null;
}
