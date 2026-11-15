using Sharplight.Tasks;

namespace Sharplight.Studio.Services.Runtime;

/// <summary>
/// A claim on the task client of a resident model (<see cref="ILoadedModel.AcquireClientAsync{TClient}"/>): while it is open the
/// model is neither evicted nor unloaded. Dispose when the run is done.
/// </summary>
/// <typeparam name="TClient">The client interface (<see cref="Sharplight.Speech.ISpeechRecognizer"/>, ...).</typeparam>
internal sealed class TaskClientLease<TClient> : IDisposable
    where TClient : class, ITaskClient
{
    private readonly IDisposable? _lease;

    /// <param name="client">The client, valid until the lease is disposed.</param>
    /// <param name="lease">What to release with the lease (the host's model lease), or null.</param>
    public TaskClientLease(TClient client, IDisposable? lease = null)
    {
        Client = client;
        _lease = lease;
    }

    /// <summary>The client.</summary>
    public TClient Client { get; }

    /// <inheritdoc/>
    public void Dispose() => _lease?.Dispose();
}
