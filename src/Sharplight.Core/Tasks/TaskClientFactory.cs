namespace Sharplight.Tasks;

/// <summary>An <see cref="ITaskClientFactory"/> made of delegates, for registering an implementation without a factory class.</summary>
/// <typeparam name="TClient">The client interface produced.</typeparam>
public sealed class TaskClientFactory<TClient> : ITaskClientFactory
    where TClient : class, ITaskClient
{
    private readonly Func<TaskModel, TClient> _create;
    private readonly Func<TaskModel, bool> _canCreate;

    /// <summary>Creates the factory.</summary>
    /// <param name="kind">The model kind served (<see cref="Models.ModelKinds"/>).</param>
    /// <param name="create">Creates a client for a model of that kind.</param>
    /// <param name="canCreate">Decides whether a model of that kind is served (e.g. by architecture), or <see langword="null"/> for all.</param>
    public TaskClientFactory(string kind, Func<TaskModel, TClient> create, Func<TaskModel, bool>? canCreate = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kind);
        ArgumentNullException.ThrowIfNull(create);
        Kind = kind;
        _create = create;
        _canCreate = canCreate ?? (_ => true);
    }

    /// <inheritdoc/>
    public string Kind { get; }

    /// <inheritdoc/>
    public Type ClientType => typeof(TClient);

    /// <inheritdoc/>
    public bool CanCreate(TaskModel model) => _canCreate(model);

    /// <inheritdoc/>
    public ITaskClient Create(TaskModel model) => _create(model);
}
