namespace Sharplight.Tasks;

/// <summary>
/// Creates task clients of one interface (<see cref="ClientType"/>) for models of one kind (<see cref="Kind"/>). Architecture packages
/// register factories as services (<c>services.AddTaskClientFactory(...)</c>) next to their adapters: a factory may serve every model of
/// its kind whose description gives it what it needs, or check the architecture in <see cref="CanCreate"/>. The
/// <see cref="TaskClientProvider"/> resolves them by kind, so adding a model kind or an implementation never changes Core types.
/// </summary>
public interface ITaskClientFactory
{
    /// <summary>The model kind (<see cref="Models.ModelKinds"/>; open-ended) whose models this factory serves.</summary>
    public string Kind { get; }

    /// <summary>The client interface this factory produces (e.g. <c>typeof(ISpeechRecognizer)</c>).</summary>
    public Type ClientType { get; }

    /// <summary>True when the factory can create a client for <paramref name="model"/> (of its <see cref="Kind"/>).</summary>
    /// <param name="model">The loaded model.</param>
    public bool CanCreate(TaskModel model);

    /// <summary>Creates a client for <paramref name="model"/>; the client does not own the model.</summary>
    /// <param name="model">The loaded model.</param>
    /// <returns>A client implementing <see cref="ClientType"/>.</returns>
    public ITaskClient Create(TaskModel model);
}
