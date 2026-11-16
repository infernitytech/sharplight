using Sharplight.Hub;

namespace Sharplight.Studio.Models;

/// <summary>Everything the user configures, persisted as JSON in the app's configuration directory.</summary>
internal sealed record AppSettings
{
    /// <summary>
    /// Where the model store is: with the other data (the storage layout's <c>models</c> folder), the system folder or
    /// <see cref="ModelsDirectory"/>. Null when never chosen: a custom folder when <see cref="ModelsDirectory"/> is set (settings of
    /// earlier versions), else the default (with the other data). <c>--models-dir</c> and <c>SHARPLIGHT_MODELS</c> take precedence.
    /// </summary>
    public ModelStoreLocation? ModelsLocation { get; init; }

    /// <summary>The custom model store directory (<see cref="ModelStoreLocation.Custom"/>).</summary>
    public string? ModelsDirectory { get; init; }

    /// <summary>The offer to move the models of a previous store (the system store, or the folder next to the app of earlier versions) was answered; it is made once.</summary>
    public bool StoreMigrationAnswered { get; init; }

    /// <summary>
    /// Hugging Face access token for gated and private repositories; null uses <c>HF_TOKEN</c>. Studio has no secret-storage
    /// abstraction (OS keyring) yet, so the token is kept in the settings file as plain text; it is sent only to huggingface.co
    /// (<c>HubRequests.Get</c> attaches it to the Hub's own host alone, never to the CDN a download redirects to).
    /// </summary>
    public string? HuggingFaceToken { get; init; }

    /// <summary>Colour theme.</summary>
    public AppTheme Theme { get; init; } = AppTheme.Dark;

    /// <summary>Key of the accent palette (see <c>AccentPalette</c>).</summary>
    public string Accent { get; init; } = "Violet";

    /// <summary>Defaults for loading models.</summary>
    public LoadSettings DefaultLoad { get; init; } = new();

    /// <summary>Defaults for new conversations.</summary>
    public SamplingSettings DefaultSampling { get; init; } = new();

    /// <summary>System prompt of new conversations.</summary>
    public string DefaultSystemPrompt { get; init; } = "You are a helpful assistant.";

    /// <summary>Let reasoning models think before answering by default.</summary>
    public bool EnableThinking { get; init; } = true;

    /// <summary>Device features never to use (provider-neutral names such as <c>matrix_f16</c>).</summary>
    public IReadOnlyList<string> DisabledFeatures { get; init; } = [];

    /// <summary>
    /// Frame the main window with the system's title bar instead of Studio's own one (which is used only where the window manager
    /// moves and resizes windows on request).
    /// </summary>
    public bool UseSystemTitleBar { get; init; }

    /// <summary>Name of the audio output device Studio plays through; null uses the system default (also when the device is gone).</summary>
    public string? AudioOutputDevice { get; init; }

    /// <summary>Name of the microphone Studio records; null uses the system default (also when the device is gone).</summary>
    public string? AudioInputDevice { get; init; }

    /// <summary>Discover lists only repositories and quantizations whose compatibility badge is <c>Supported</c>.</summary>
    public bool ShowOnlySupported { get; init; }

    /// <summary>The filters and sort order of Discover.</summary>
    public DiscoverFilterSettings Discover { get; init; } = new();

    /// <summary>Compile the GPU kernels into a package right after it was downloaded and converted.</summary>
    public bool PrecompileAfterDownload { get; init; } = true;

    /// <summary>
    /// Keep compiled kernels and driver pipeline caches in the per-user compilation cache, so models that are not precompiled load fast
    /// from the second time on. Off: nothing is read from or written to the cache.
    /// </summary>
    public bool UseCompilationCache { get; init; } = true;

    /// <summary>Keep the downloaded source files next to the converted package.</summary>
    public bool KeepSourceFiles { get; init; }

    /// <summary>Local model files and packages added to the library (outside the store).</summary>
    public IReadOnlyList<string> LocalModels { get; init; } = [];

    /// <summary>Sampling presets saved by the user (the built-in ones are not stored).</summary>
    public IReadOnlyList<SamplingPreset> Presets { get; init; } = [];

    /// <summary>Memory reserve, resident model limit, idle unloading and on-demand loading.</summary>
    public ResidencySettings Residency { get; init; } = new();

    /// <summary>Endpoint settings.</summary>
    public ServerSettings Server { get; init; } = new();

    /// <summary>Downloads not finished when the app closed; running ones resume at the next start, paused ones stay paused.</summary>
    public IReadOnlyList<PendingDownload> Downloads { get; init; } = [];

    /// <summary>Hide the welcome hints after the first model was downloaded.</summary>
    public bool OnboardingDone { get; init; }

    /// <summary>Check for a new Studio version at startup and every few hours (installed builds only).</summary>
    public bool AutoCheckForUpdates { get; init; } = true;

    /// <summary>When the last update check finished (successfully or not); null when none ran.</summary>
    public DateTimeOffset? LastUpdateCheck { get; init; }

    /// <summary>
    /// Update feed URL that replaces the one baked into the build; null keeps it. The <c>SHARPLIGHT_STUDIO_UPDATE_URL</c>
    /// environment variable still takes precedence.
    /// </summary>
    public string? UpdateUrl { get; init; }
}
