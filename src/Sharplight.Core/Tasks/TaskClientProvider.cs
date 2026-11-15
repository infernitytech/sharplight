namespace Sharplight.Tasks;

/// <summary>
/// Creates task clients for loaded models from the registered <see cref="ITaskClientFactory"/> services: the factory for the model's
/// kind (<see cref="TaskModel.Description"/>) that produces the requested client interface and accepts the model. When several match,
/// the last registered wins, so a host or architecture package can override a generic implementation. Registered as a singleton by
/// <c>services.AddSharplightTasks()</c>; thread-safe.
/// </summary>
public sealed class TaskClientProvider
{
    private readonly ITaskClientFactory[] _factories;

    /// <summary>Creates the provider over the registered factories.</summary>
    /// <param name="factories">The factories, in registration order.</param>
    public TaskClientProvider(IEnumerable<ITaskClientFactory> factories)
    {
        ArgumentNullException.ThrowIfNull(factories);
        _factories = [.. factories];
    }

    /// <summary>The model kinds at least one factory serves.</summary>
    public IReadOnlyCollection<string> Kinds => [.. _factories.Select(f => f.Kind).Distinct(StringComparer.Ordinal)];

    /// <summary>True when a registered factory creates a <typeparamref name="TClient"/> for <paramref name="model"/>.</summary>
    /// <typeparam name="TClient">The client interface.</typeparam>
    /// <param name="model">The loaded model.</param>
    public bool CanCreate<TClient>(TaskModel model)
        where TClient : class, ITaskClient
        => Find(typeof(TClient), model) is not null;

    /// <summary>Creates a <typeparamref name="TClient"/> for <paramref name="model"/>; the client does not own the model.</summary>
    /// <typeparam name="TClient">The client interface, e.g. <c>ISpeechRecognizer</c>.</typeparam>
    /// <param name="model">The loaded model.</param>
    /// <exception cref="NotSupportedException">No registered factory serves the model's kind and architecture with this client interface.</exception>
    public TClient Create<TClient>(TaskModel model)
        where TClient : class, ITaskClient
    {
        ArgumentNullException.ThrowIfNull(model);
        var factory = Find(typeof(TClient), model)
            ?? throw new NotSupportedException($"No {typeof(TClient).Name} is registered for '{model.Description.Kind}' models of architecture "
                + $"'{model.Definition.Architecture}'. Register one with services.AddTaskClientFactory(...) or services.AddSharplight().");
        var client = factory.Create(model);
        if (client is TClient typed)
        {
            return typed;
        }

        client.Dispose();
        throw new InvalidOperationException($"The {factory.Kind} factory for {typeof(TClient).Name} produced a {client.GetType().Name}.");
    }

    private ITaskClientFactory? Find(Type clientType, TaskModel model)
    {
        for (int i = _factories.Length - 1; i >= 0; i--)
        {
            var factory = _factories[i];
            if (factory.Kind == model.Description.Kind && clientType.IsAssignableFrom(factory.ClientType) && factory.CanCreate(model))
            {
                return factory;
            }
        }

        return null;
    }
}
