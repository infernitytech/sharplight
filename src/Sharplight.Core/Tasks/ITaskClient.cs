namespace Sharplight.Tasks;

/// <summary>
/// The common shape of every task client (text generation, speech recognition, speech synthesis, image generation, and kinds added
/// later): created for a loaded <see cref="TaskModel"/> by an <see cref="ITaskClientFactory"/>, it owns its compiled graphs and
/// device state (not the model) and releases them on <see cref="IDisposable.Dispose"/>. Requests are asynchronous, take an options
/// object and a <see cref="CancellationToken"/>, and come in a whole-result and a streaming form.
/// <para>
/// Mirrors the client pattern of Microsoft.Extensions.AI (<c>IChatClient</c>, <c>ISpeechToTextClient</c>, <c>IImageGenerator</c>):
/// metadata, <see cref="GetService"/> for the underlying objects, options per request. Sharplight.Extensions.AI adapts the clients
/// to those interfaces. Clients are not thread-safe unless they say so: run one request at a time per client.
/// </para>
/// </summary>
public interface ITaskClient : IDisposable
{
    /// <summary>What the client runs: its model kind, model name and architecture.</summary>
    public TaskClientMetadata Metadata { get; }

    /// <summary>
    /// Returns an object of <paramref name="serviceType"/> that the client holds (e.g. the <see cref="TaskModel"/>, or the
    /// <c>LlmModel</c> of a text generator), or <see langword="null"/>.
    /// </summary>
    /// <param name="serviceType">The requested type.</param>
    /// <param name="serviceKey">An optional key; implementations without keyed services return <see langword="null"/> for any key.</param>
    public object? GetService(Type serviceType, object? serviceKey = null);
}
