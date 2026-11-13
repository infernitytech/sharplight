namespace Sharplight.Llm.Serving.Hosting;

/// <summary>A claim on a loaded model: while any lease is open the model is neither evicted nor unloaded. Dispose when the request (incl. its streamed response) is done.</summary>
public sealed class ModelLease : IDisposable
{
    private readonly Action _release;
    private int _released;
    internal ModelLease(string name, LoadedModel model, Action release) { Name = name; Model = model; _release = release; }
    public string Name { get; }
    public LoadedModel Model { get; }
    public T As<T>() where T : LoadedModel => Model as T ?? throw new InvalidOperationException($"Model '{Name}' is a {Model.GetType().Name}, not a {typeof(T).Name}.");
    public void Dispose() { if (Interlocked.Exchange(ref _released, 1) == 0)
        {
            _release();
        }
    }
}
