using Sharplight.Tasks;

namespace Sharplight.Hosting.Loaders;

/// <summary>A loaded model of kind <c>Task</c>: what a lease on it gives access to.</summary>
public interface ITaskClientModel
{
    /// <summary>The client of type <typeparamref name="TClient"/>, or <see langword="null"/> when the model has none; valid while the lease is held.</summary>
    public TClient? FindClient<TClient>()
        where TClient : class, ITaskClient;
}
