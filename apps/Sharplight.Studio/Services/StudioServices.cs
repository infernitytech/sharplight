using Sharplight.Studio.Services.Documents;
using Sharplight.Studio.Services.Hub;
using Sharplight.Studio.Services.Library;
using Sharplight.Studio.Services.Monitoring;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Precompile;
using Sharplight.Studio.Services.Runtime;
using Sharplight.Studio.Services.Server;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.Services.Storage;
using Sharplight.Studio.Services.Updates;

namespace Sharplight.Studio.Services;

/// <summary>Every service the view models use; the app builds the real ones, tests pass fakes.</summary>
internal sealed record StudioServices(
    ISettingsService Settings,
    IModelRuntime Runtime,
    IHubService Hub,
    IModelLibrary Library,
    IConversationStore Conversations,
    IStatsStore Stats,
    IApiServer Server,
    IDocumentExtractor Documents,
    ISystemMonitor Monitor,
    IPlatformServices Platform,
    IUiDispatcher Dispatcher,
    IThemeService Theme,
    IUpdateService Updates,
    IPrecompileService Precompiler);
