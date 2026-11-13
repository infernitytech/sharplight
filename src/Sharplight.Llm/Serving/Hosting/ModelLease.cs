namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A claim on a loaded model: while any lease is open the model is neither evicted nor unloaded. Dispose when the request (incl. its streamed response) is done.</summary>
public sealed class ModelLease : IDisposable
{
    private readonly Action _release;
    private int _released;
    internal ModelLease(string name, LoadedModel model, Action release)
    {
        Name = name;
        Model = model;
        _release = release;
    }
    /// <summary>The registered model name.</summary>
    public string Name { get; }
    /// <summary>The loaded model, valid until the lease is disposed.</summary>
    public LoadedModel Model { get; }
    /// <summary>The model as <typeparamref name="T"/>.</summary>
    /// <exception cref="InvalidOperationException">The model is not a <typeparamref name="T"/>.</exception>
    public T As<T>() where T : LoadedModel => Model as T ?? throw new InvalidOperationException($"Model '{Name}' is a {Model.GetType().Name}, not a {typeof(T).Name}.");
    /// <summary>Releases the lease (idempotent); the model may then be evicted or unloaded.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _released, 1) == 0)
        {
            _release();
        }
    }
}
