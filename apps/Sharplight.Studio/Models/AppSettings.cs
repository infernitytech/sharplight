namespace Sharplight.Studio.Models;

/// <summary>Everything the user configures, persisted as JSON in the app's configuration directory.</summary>
internal sealed record AppSettings
{
    /// <summary>The model store; null resolves like the CLI (<c>--models-dir</c>, <c>SHARPLIGHT_MODELS</c>, portable, per user).</summary>
    public string? ModelsDirectory { get; init; }

    /// <summary>Hugging Face access token for gated and private repositories; null uses <c>HF_TOKEN</c>.</summary>
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

    /// <summary>Compile the GPU kernels into a package right after it was downloaded and converted.</summary>
    public bool PrecompileAfterDownload { get; init; } = true;

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
}
