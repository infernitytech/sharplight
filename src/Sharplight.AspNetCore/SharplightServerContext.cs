using Microsoft.AspNetCore.Http;
using Sharplight.Llm.Serving;
using Sharplight.Llm.Serving.Chat;
using Sharplight.Llm.Serving.Hosting;

namespace Sharplight.AspNetCore;

/// <summary>
/// Shared state behind the endpoints (one per host): the models (a <see cref="ModelHost"/>) and the server options. The per-request
/// methods apply the <see cref="SharplightRequestContext"/> of the request (visible models and aliases, defaults, limits, tools,
/// scope) when the request went through the Sharplight pipeline, else the server-wide options.
/// </summary>
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

    /// <summary>True when the endpoints expose <paramref name="name"/> (<see cref="SharplightServerOptions.ModelFilter"/>).</summary>
    public bool IsServed(string name) => Options.ModelFilter?.Invoke(name) ?? true;

    /// <summary>The registered models the endpoints expose (server-wide view).</summary>
    public IReadOnlyList<ModelInfo> ListModels() => Options.ModelFilter is null ? Host.List() : [.. Host.List().Where(m => IsServed(m.Name))];

    /// <summary>The models this request may see, under the names it uses (aliases).</summary>
    public IReadOnlyList<ModelInfo> ListModels(HttpContext http)
    {
        if (Request(http)?.Models is not { } aliases)
        {
            return ListModels();
        }

        return [.. aliases.Select(alias => Describe(alias.Key, alias.Value))];
    }

    /// <summary>One model this request may see, by the name it uses, or null.</summary>
    public ModelInfo? FindModel(HttpContext http, string name)
    {
        if (Request(http)?.Models is { } aliases)
        {
            return aliases.TryGetValue(name, out var target) ? Describe(name, target) : null;
        }

        return IsServed(name) ? Host.Get(name) : null;
    }

    /// <summary>The first exposed model of <paramref name="kind"/>, used when a request names none.</summary>
    public string? DefaultName(ModelKind kind = ModelKind.Chat)
        => Options.ModelFilter is null ? Host.DefaultName(kind) : ListModels().FirstOrDefault(m => m.Kind == kind)?.Name;

    /// <summary>
    /// Resolves the request's <c>model</c> (null/empty = the default model of <paramref name="kind"/>), loading it when needed, admits
    /// the request (<see cref="IRequestAdmission"/>) and returns a lease that keeps the model loaded until disposed (dispose after the
    /// response is written). Unknown names, and names the request's context does not allow, raise a 404 in the API's error shape
    /// (unless <see cref="SharplightServerOptions.UnknownModelFallsBackToDefault"/> without per-request models).
    /// </summary>
    public async Task<ModelLease> AcquireAsync(HttpContext http, string? requested, ModelKind kind = ModelKind.Chat)
    {
        var feature = SharplightRequestFeature.Of(http);
        var (alias, name) = Resolve(http, feature?.Context, requested, kind);
        ModelLease lease;
        try
        {
            lease = await Host.AcquireAsync(name, kind, http.RequestAborted);
        }
        catch (ModelNotFoundException ex)
        {
            throw ApiException.ModelNotFound(ex.Message, ApiErrors.IsAnthropic(http));
        }
        catch (ModelKindMismatchException ex)
        {
            throw ApiException.ModelNotFound(ex.Message, ApiErrors.IsAnthropic(http));
        }

        try
        {
            await AdmitAsync(http, alias);
        }
        catch
        {
            lease.Dispose();
            throw;
        }

        return lease;
    }

    /// <summary>Records the model of the request and runs the admission hook (<see cref="IRequestAdmission"/>) for it.</summary>
    public static async Task AdmitAsync(HttpContext http, string alias)
    {
        if (SharplightRequestFeature.Of(http) is not { } feature)
        {
            return;
        }

        feature.Model = alias;
        if (feature.Admission is { } admission && await admission.AdmitAsync(feature.Context, alias, http, http.RequestAborted) is { } handle)
        {
            feature.Hold(handle);
        }
    }

    /// <summary>The model name responses report: the alias the client used.</summary>
    public static string ResponseName(HttpContext http, ModelLease lease) => SharplightRequestFeature.Of(http)?.Model ?? lease.Name;

    /// <summary>Sampling defaults of the request.</summary>
    public SamplingParams DefaultSampling(HttpContext http) => Defaults(http)?.Sampling ?? Options.DefaultSampling;

    /// <summary>The request's time budget.</summary>
    public TimeSpan? RequestTimeout(HttpContext http) => Defaults(http)?.RequestTimeout ?? Options.RequestTimeout;

    /// <summary>
    /// Notes the model the request names (or the default alias) so the per-model defaults of the request's context
    /// (<see cref="SharplightRequestContext.ModelDefaults"/>) apply while the body is parsed.
    /// </summary>
    public void NoteModel(HttpContext http, string? requested)
    {
        if (SharplightRequestFeature.Of(http) is not { } feature)
        {
            return;
        }

        var context = feature.Context;
        feature.RequestedModel = !string.IsNullOrEmpty(requested)
            ? requested
            : context.DefaultModel ?? context.Models?.FirstOrDefault(entry => (Host.Get(entry.Value)?.Kind ?? ModelKind.Chat) == ModelKind.Chat).Key ?? DefaultName();
    }

    /// <summary>The effective defaults of the request: the per-model defaults of its model over the context's defaults.</summary>
    public static SharplightRequestDefaults? Defaults(HttpContext http)
    {
        if (SharplightRequestFeature.Of(http) is not { } feature)
        {
            return null;
        }

        var context = feature.Context;
        return feature.RequestedModel is { } model && context.ModelDefaults?.TryGetValue(model, out var own) == true
            ? own.Over(context.Defaults)
            : context.Defaults;
    }

    /// <summary>The messages with the default system prompt first when they have no system message.</summary>
    public static IReadOnlyList<ChatTurn> WithSystemPrompt(HttpContext http, IReadOnlyList<ChatTurn> messages)
    {
        if (Defaults(http)?.SystemPrompt is not { } system || messages.Any(message => message.Role == "system"))
        {
            return messages;
        }

        return [new ChatTurn("system", system), .. messages];
    }

    /// <summary>The generated-token limit: the request's value, else the context's default, else <paramref name="fallback"/>; cut to the context's limit (0 = unlimited).</summary>
    public static int MaxTokens(HttpContext http, int? requested, int fallback)
    {
        var defaults = Defaults(http);
        int value = requested ?? defaults?.MaxTokens ?? fallback;
        return defaults?.MaxTokensLimit is int limit && (value <= 0 || value > limit) ? limit : value;
    }

    /// <summary>The fair-share tenant of the request (the context's, else the end-user id or key within the scope).</summary>
    public static string? TenantOf(HttpContext http, string? user)
    {
        var context = Request(http);
        if (context?.Tenant is { } tenant)
        {
            return tenant;
        }

        var own = Tenants.Of(http, user);
        return context?.Scope is { } scope ? $"{scope}/{own ?? context.Principal ?? ""}" : own;
    }

    /// <summary>The prefix-cache partition of the request (its scope).</summary>
    public static string? CachePartition(HttpContext http) => Request(http)?.Scope;

    /// <summary>
    /// The prefix-cache salt of the request: the context's <see cref="SharplightRequestContext.CacheSalt"/>, followed by the client's
    /// <c>cache_salt</c> when <see cref="SharplightServerOptions.AllowClientCacheSalt"/> allows it.
    /// </summary>
    /// <exception cref="ApiException">The client sent a salt the server does not accept (400).</exception>
    public string? CacheSalt(HttpContext http, string? client)
    {
        var host = Request(http)?.CacheSalt;
        if (client is null)
        {
            return host;
        }

        if (!Options.AllowClientCacheSalt)
        {
            throw new ApiException(400, "invalid_request_error", "cache_salt is not accepted by this server.", null, "cache_salt");
        }

        return host is null ? "client:" + client : host + "\0client:" + client;
    }

    /// <summary>Refuses tools the request's context does not allow (400).</summary>
    public static void CheckTools(HttpContext http, IEnumerable<string> names)
    {
        if (Request(http)?.AllowedTools is not { } allowed)
        {
            return;
        }

        foreach (var name in names.Where(name => !allowed.Contains(name)))
        {
            throw new ApiException(400, "invalid_request_error", $"The tool '{name}' is not allowed on this endpoint.", "tool_not_allowed", "tools");
        }
    }

    /// <summary>Chat service of a lease from <see cref="AcquireAsync"/>.</summary>
    public static ModelService ChatOf(ModelLease lease) => lease.As<LoadedModel>() is IChatModel c ? c.Service : throw new InvalidOperationException($"Model '{lease.Name}' is not a chat model.");

    /// <summary>The alias map of the request's context, or null when the request may use every served model.</summary>
    public static IReadOnlyDictionary<string, string>? Aliases(HttpContext http, out string? defaultModel)
    {
        var context = Request(http);
        defaultModel = context?.DefaultModel;
        return context?.Models;
    }

    private static SharplightRequestContext? Request(HttpContext http) => SharplightRequestFeature.Of(http)?.Context;

    private (string Alias, string Name) Resolve(HttpContext http, SharplightRequestContext? context, string? requested, ModelKind kind)
    {
        bool anthropic = ApiErrors.IsAnthropic(http);
        if (context?.Models is { } aliases)
        {
            string? alias = string.IsNullOrEmpty(requested)
                ? context.DefaultModel ?? aliases.FirstOrDefault(entry => (Host.Get(entry.Value)?.Kind ?? kind) == kind).Key
                : requested;
            if (alias is null || !aliases.TryGetValue(alias, out var target))
            {
                throw ApiException.ModelNotFound(alias is null ? $"No {kind.ToString().ToLowerInvariant()} model is available." : $"The model '{alias}' does not exist.", anthropic);
            }

            return (alias, target);
        }

        string? name = requested;
        if (string.IsNullOrEmpty(name) || (Options.UnknownModelFallsBackToDefault && !(Host.Contains(name) && IsServed(name))))
        {
            name = DefaultName(kind);
        }

        if (name is null)
        {
            throw ApiException.ModelNotFound($"No {kind.ToString().ToLowerInvariant()} model is available.", anthropic);
        }

        if (!IsServed(name))
        {
            throw ApiException.ModelNotFound($"The model '{name}' does not exist.", anthropic);
        }

        return (name, name);
    }

    /// <summary>The host's view of <paramref name="target"/> under <paramref name="alias"/> (an on-demand model not registered yet shows as unloaded).</summary>
    private ModelInfo Describe(string alias, string target)
        => Host.Get(target) is { } info
            ? info with { Name = alias }
            : new ModelInfo(alias, ModelKind.Chat, ModelState.Unloaded, false, false, 0, 0, 0, null, null, null, null);
}
