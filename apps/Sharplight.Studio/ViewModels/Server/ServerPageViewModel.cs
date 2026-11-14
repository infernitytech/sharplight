using System.Collections.ObjectModel;
using System.Security.Cryptography;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Server;

namespace Sharplight.Studio.ViewModels.Server;

/// <summary>The server page: endpoint settings, the served models, start / stop and the request log.</summary>
internal sealed partial class ServerPageViewModel : PageViewModel
{
    private const int MaxLogRows = 500;

    private readonly StudioServices _services;
    private readonly bool _initializing;

    public ServerPageViewModel(StudioServices services)
    {
        _services = services;
        _initializing = true;
        var settings = services.Settings.Current.Server;
        Host = settings.Host;
        Port = settings.Port;
        ApiKey = settings.ApiKey;
        CorsOrigins = settings.CorsOrigins;
        OpenAi = settings.OpenAi;
        Anthropic = settings.Anthropic;
        LoadOnDemand = services.Settings.Current.Residency.LoadOnDemand;
        _initializing = false;
        services.Server.RequestLogged += (_, entry) => services.Dispatcher.Post(() => AddLog(entry));
        services.Runtime.ModelsChanged += (_, _) => services.Dispatcher.Post(RefreshModels);
        services.Server.StateChanged += (_, _) => services.Dispatcher.Post(SyncState);
        RefreshModels();
        SyncState();
    }

    /// <inheritdoc/>
    public override string Title => "Server";

    /// <inheritdoc/>
    public override string Icon => "IconServer";

    /// <summary>Loaded models with their served switch.</summary>
    public ObservableCollection<ServedModelViewModel> Models { get; } = [];

    /// <summary>Answered requests, newest first.</summary>
    public ObservableCollection<HttpLogRowViewModel> Log { get; } = [];

    /// <summary>Listen address.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PreviewUrl), nameof(CurlExample))]
    public partial string Host { get; set; }

    /// <summary>Port.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(PreviewUrl), nameof(CurlExample))]
    public partial double Port { get; set; }

    /// <summary>API key (empty = none).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CurlExample))]
    public partial string ApiKey { get; set; }

    /// <summary>CORS origins.</summary>
    [ObservableProperty]
    public partial string CorsOrigins { get; set; }

    /// <summary>OpenAI API on.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Endpoints))]
    public partial bool OpenAi { get; set; }

    /// <summary>Anthropic API on.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(Endpoints))]
    public partial bool Anthropic { get; set; }

    /// <summary>Listening.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(StatusText))]
    public partial bool IsRunning { get; set; }

    /// <summary>Starting or stopping.</summary>
    [ObservableProperty]
    public partial bool IsBusy { get; set; }

    /// <summary>Error of the last start.</summary>
    [ObservableProperty]
    public partial string? Error { get; set; }

    /// <summary>The address while running.</summary>
    [ObservableProperty]
    public partial string? BaseUrl { get; set; }

    /// <summary>Requests answered since the start.</summary>
    [ObservableProperty]
    public partial int RequestCount { get; set; }

    /// <summary>A request for a library model that is not resident loads it.</summary>
    [ObservableProperty]
    public partial bool LoadOnDemand { get; set; }

    /// <summary>Status line.</summary>
    public string StatusText => !IsRunning
        ? "Stopped"
        : Models.All(model => model.IsServed)
            ? LoadOnDemand ? "Serving every resident model, library models on demand" : "Serving every resident model"
            : $"Serving {string.Join(", ", Models.Where(model => model.IsServed).Select(model => model.Id))}";

    /// <summary>The address the settings describe.</summary>
    public string PreviewUrl => $"http://{Host}:{(int)Port}";

    /// <summary>True when a model is loaded.</summary>
    public bool HasModels => Models.Count > 0;

    /// <summary>Endpoints the enabled APIs map.</summary>
    public IReadOnlyList<EndpointInfo> Endpoints
    {
        get
        {
            var list = new List<EndpointInfo>();
            if (OpenAi)
            {
                list.Add(new("POST", "/v1/chat/completions", "OpenAI chat (streaming, tools, JSON schema)"));
                list.Add(new("POST", "/v1/completions", "OpenAI text completion"));
            }

            if (Anthropic)
            {
                list.Add(new("POST", "/v1/messages", "Anthropic Messages (streaming, tools, thinking)"));
                list.Add(new("POST", "/v1/messages/count_tokens", "Anthropic token count"));
            }

            list.Add(new("GET", "/v1/models", "Served models"));
            list.Add(new("GET", "/health", "Status"));
            list.Add(new("GET", "/metrics", "Prometheus metrics"));
            return list;
        }
    }

    /// <summary>A ready-to-paste request.</summary>
    public string CurlExample
    {
        get
        {
            var model = Models.FirstOrDefault(candidate => candidate.IsServed)?.Id ?? "MODEL";
            var auth = ApiKey.Trim().Length > 0 ? $" \\\n  -H \"Authorization: Bearer {ApiKey.Trim()}\"" : "";
            return $"curl {PreviewUrl}/v1/chat/completions \\\n  -H \"Content-Type: application/json\"{auth} \\\n"
                + $"  -d '{{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}], \"stream\": true}}'";
        }
    }

    partial void OnHostChanged(string value) => Save();

    partial void OnPortChanged(double value) => Save();

    partial void OnApiKeyChanged(string value) => Save();

    partial void OnCorsOriginsChanged(string value) => Save();

    partial void OnOpenAiChanged(bool value) => Save();

    partial void OnAnthropicChanged(bool value) => Save();

    partial void OnLoadOnDemandChanged(bool value)
    {
        if (!_initializing)
        {
            _services.Settings.Update(current => current with { Residency = current.Residency with { LoadOnDemand = value } });
            OnPropertyChanged(nameof(StatusText));
        }
    }

    [RelayCommand]
    private async Task StartAsync()
    {
        IsBusy = true;
        Error = null;
        try
        {
            Save();
            await _services.Server.StartAsync(_services.Settings.Current.Server, CancellationToken.None);
            RequestCount = 0;
        }
        catch (Exception error) when (error is InvalidOperationException or IOException)
        {
            Error = error.Message;
        }
        finally
        {
            IsBusy = false;
            SyncState();
        }
    }

    [RelayCommand]
    private async Task StopAsync()
    {
        IsBusy = true;
        try
        {
            await _services.Server.StopAsync();
        }
        finally
        {
            IsBusy = false;
            SyncState();
        }
    }

    [RelayCommand]
    private void GenerateKey()
    {
        ApiKey = "sk-sharplight-" + Convert.ToHexString(RandomNumberGenerator.GetBytes(16)).ToLowerInvariant();
    }

    [RelayCommand]
    private Task CopyAsync(string text) => _services.Platform.CopyTextAsync(text);

    [RelayCommand]
    private void ClearLog() => Log.Clear();

    private void AddLog(HttpLogEntry entry)
    {
        Log.Insert(0, new HttpLogRowViewModel(entry));
        if (Log.Count > MaxLogRows)
        {
            Log.RemoveAt(Log.Count - 1);
        }

        RequestCount++;
    }

    private void RefreshModels()
    {
        // Requests route to resident models by name, so the running server picks up loads and unloads without a restart.
        var served = _services.Settings.Current.Server.ServedModels;
        var current = _services.Runtime.LoadedModels;
        if (Models.Select(item => item.Model).SequenceEqual(current))
        {
            foreach (var item in Models)
            {
                item.Refresh();
            }

            return;
        }

        Models.Clear();
        foreach (var model in current)
        {
            Models.Add(new ServedModelViewModel(model, served.Count == 0 || served.Contains(model.Id), OnServedChanged));
        }

        OnPropertyChanged(nameof(HasModels));
        OnPropertyChanged(nameof(CurlExample));
        OnPropertyChanged(nameof(StatusText));
    }

    private void OnServedChanged()
    {
        Save();
        OnPropertyChanged(nameof(StatusText));
        if (IsRunning)
        {
            _ = StartAsync();   // the filter is fixed per start
        }
    }

    private void SyncState()
    {
        IsRunning = _services.Server.IsRunning;
        BaseUrl = _services.Server.BaseUrl;
        OnPropertyChanged(nameof(StatusText));
    }

    private void Save()
    {
        if (_initializing)
        {
            return;
        }

        var servedIds = Models.Count > 0 && Models.All(model => model.IsServed) ? [] : Models.Where(model => model.IsServed).Select(model => model.Id).ToList();
        var settings = new ServerSettings
        {
            Host = Host.Trim(),
            Port = Math.Clamp((int)Port, 1, 65535),
            ApiKey = ApiKey,
            CorsOrigins = CorsOrigins,
            OpenAi = OpenAi,
            Anthropic = Anthropic,
            ServedModels = servedIds,
        };
        _services.Settings.Update(current => current with { Server = settings });
        _services.Server.SetServedModels(servedIds);
        OnPropertyChanged(nameof(CurlExample));
    }
}
