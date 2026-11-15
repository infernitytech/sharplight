using System.Collections.Concurrent;

namespace Sharplight.Tasks;

/// <summary>
/// Registry of task utility factories by task id and, optionally, canonical architecture. Architecture packages register the
/// model-specific host logic (e.g. a Whisper recognizer, a Stable Diffusion pipeline) next to their adapters; a generic
/// implementation registered without an architecture serves every model of that task whose descriptor gives it what it needs.
/// Lookups go by the architecture id stored in the definition, so they work for models loaded from saved packages.
/// </summary>
public static class ModelTaskRegistry
{
    private const string AnyArchitecture = "";

    private static readonly ConcurrentDictionary<(string Kind, string Architecture), Func<TaskModel, object>> s_factories = new();

    /// <summary>
    /// Registers a factory; a later registration for the same task and architecture replaces the earlier one.
    /// </summary>
    /// <param name="kind">The task id (<see cref="ModelTasks"/>; open-ended).</param>
    /// <param name="factory">Creates the task utility (e.g. an <see cref="Speech.ISpeechRecognizer"/>) for a loaded model.</param>
    /// <param name="architecture">The canonical architecture the factory is specific to, or <see langword="null"/> for a generic one.</param>
    public static void Register(string kind, Func<TaskModel, object> factory, string? architecture = null)
        => s_factories[(kind, architecture ?? AnyArchitecture)] = factory;

    /// <summary>True when a factory for <paramref name="kind"/> serves <paramref name="architecture"/> (specific or generic).</summary>
    public static bool Supports(string kind, string architecture)
        => s_factories.ContainsKey((kind, architecture)) || s_factories.ContainsKey((kind, AnyArchitecture));

    /// <summary>
    /// Creates the task utility for a model: the factory registered for the task and the model's architecture, else the generic one.
    /// </summary>
    /// <typeparam name="T">The task interface, e.g. <see cref="Speech.ISpeechRecognizer"/>.</typeparam>
    /// <param name="model">The loaded model.</param>
    /// <param name="kind">The task id; defaults to the kind of the model's descriptor.</param>
    /// <exception cref="NotSupportedException">No factory is registered, or the factory's product does not implement <typeparamref name="T"/>.</exception>
    public static T Create<T>(TaskModel model, string? kind = null) where T : class
    {
        string architecture = model.Definition.Architecture;
        kind ??= model.Task?.Kind ?? throw new NotSupportedException($"{model.Definition.Name} ({architecture}) declares no task; pass the task id.");
        if (!s_factories.TryGetValue((kind, architecture), out var factory) && !s_factories.TryGetValue((kind, AnyArchitecture), out factory))
        {
            throw new NotSupportedException($"No '{kind}' implementation is registered for architecture '{architecture}'. Hosts register them with "
                + "SharplightDefaults.Register() (Sharplight.Defaults package) or the Register methods of the architecture packages.");
        }

        object created = factory(model);
        if (created is T utility)
        {
            return utility;
        }

        (created as IDisposable)?.Dispose();
        throw new NotSupportedException($"The '{kind}' implementation for '{architecture}' does not implement {typeof(T).Name}.");
    }
}
