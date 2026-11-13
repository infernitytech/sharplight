using Microsoft.AspNetCore.Http;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.Server;

/// <summary>Shared state behind the endpoints (one per host): the models (a <see cref="ModelHost"/>) and the server options.</summary>
internal sealed class SharplightServerContext
{
    public SharplightServerContext(ModelHost host, SharplightServerOptions options)
    {
        Host = host;
        Options = options;
    }

    /// <summary>Single-model form: the service is registered as the (pinned, never reloaded) default model. The caller keeps ownership of its engine.</summary>
    public SharplightServerContext(ModelService model, SharplightServerOptions options)
    {
        Host = new ModelHost();
        Host.Adopt(model.ModelName, ChatModelInstance.Wrap(model));
        Options = options;
        Options.UnknownModelFallsBackToDefault = true;
    }

    public ModelHost Host { get; }
    public SharplightServerOptions Options { get; }
    internal bool ModelsMapped;

    /// <summary>The default chat model's service (loads it if needed); for code written for a single model. Prefer <see cref="AcquireAsync"/>.</summary>
    public ModelService Model
    {
        get
        {
            var name = Host.DefaultName() ?? throw new InvalidOperationException("No chat model is registered.");
            using var lease = Host.AcquireAsync(name).GetAwaiter().GetResult();
            return ChatOf(lease);   // valid until the model is unloaded; hold a lease for long work
        }
    }

    /// <summary>
    /// Resolves the request's <c>model</c> (null/empty = the default model of <paramref name="kind"/>), loading it when needed, and
    /// returns a lease that keeps it loaded until disposed (dispose after the response is written). Unknown names raise a 404 in the
    /// API's error shape, unless <see cref="SharplightServerOptions.UnknownModelFallsBackToDefault"/>.
    /// </summary>
    public async Task<ModelLease> AcquireAsync(HttpContext http, string? requested, ModelKind kind = ModelKind.Chat)
    {
        string? name = requested;
        if (string.IsNullOrEmpty(name) || (Options.UnknownModelFallsBackToDefault && !Host.Contains(name)))
        {
            name = Host.DefaultName(kind);
        }

        if (name is null)
        {
            throw new ApiException(404, ApiErrors.IsAnthropic(http) ? "not_found_error" : "invalid_request_error", $"No {kind.ToString().ToLowerInvariant()} model is available.", "model_not_found", "model");
        }

        try { return await Host.AcquireAsync(name, kind, http.RequestAborted); }
        catch (ModelNotFoundException ex) { throw new ApiException(404, ApiErrors.IsAnthropic(http) ? "not_found_error" : "invalid_request_error", ex.Message, "model_not_found", "model"); }
        catch (ModelKindMismatchException ex) { throw new ApiException(404, ApiErrors.IsAnthropic(http) ? "not_found_error" : "invalid_request_error", ex.Message, "model_not_found", "model"); }
    }

    /// <summary>Chat service of a lease from <see cref="AcquireAsync"/>.</summary>
    public static ModelService ChatOf(ModelLease lease) => lease.As<LoadedModel>() is IChatModel c ? c.Service : throw new InvalidOperationException($"Model '{lease.Name}' is not a chat model.");
}
