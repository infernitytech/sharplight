using Avalonia;
using Avalonia.Headless;
using Avalonia.Logging;
using Avalonia.Media.Imaging;
using Avalonia.Threading;
using Avalonia.VisualTree;
using Sharplight.Models;
using Sharplight.Runtime;
using Sharplight.Studio.Models;
using Sharplight.Studio.Tests.FirstRun;
using Sharplight.Studio.Platform;
using Sharplight.Studio.ViewModels;
using Sharplight.Studio.ViewModels.Models;
using Sharplight.Studio.ViewModels.Settings;
using Sharplight.Studio.Views;

namespace Sharplight.Studio.Tests;

/// <summary>
/// One screenshot test per page and per Settings tab and Discover source, in the light and the dark theme at 1440 and 900 pixels wide. The pages
/// are filled with demo data, and each test checks that the page binds without errors and that no control is clipped or painted over its
/// neighbours (<see cref="LayoutChecker"/>), then saves the frame as a PNG. There are no golden images: the PNGs are test artifacts that CI
/// uploads (<c>studio.yml</c>) for a human to look at, into <c>SHARPLIGHT_STUDIO_SCREENSHOTS</c> (default: <c>page-screenshots</c> in the test output).
/// </summary>
public sealed class PageScreenshotTests
{
    private static readonly (double Width, double Height)[] s_sizes = [(1440, 900), (900, 600)];

    private static readonly (string Name, Func<MainWindowViewModel, MainWindow, Task> Show)[] s_pages =
    [
        ("chat", (model, _) => model.NavigateAsync(model.Chat)),
        ("welcome", async (model, window) =>
        {
            await model.NavigateAsync(model.Welcome);
            await model.Welcome.Loading;
            await HeadlessApp.WaitForAsync(window, () => model.Welcome.HasItems);
        }),
        ("chat-get-model", async (model, _) =>
        {
            await model.NavigateAsync(model.Chat);
            model.Chat.NewChatCommand.Execute(null);
        }),
        ("transcribe", (model, _) => model.NavigateAsync(model.Transcribe)),
        ("speak", (model, _) => model.NavigateAsync(model.Speak)),
        ("search", (model, _) => model.NavigateAsync(model.Search)),
        ("audio", (model, _) => model.NavigateAsync(model.Audio)),
        ("image", (model, _) => model.NavigateAsync(model.Image)),
        ("video", (model, _) => model.NavigateAsync(model.Video)),
        ("vision", (model, _) => model.NavigateAsync(model.Vision)),
        ("enhance", (model, _) => model.NavigateAsync(model.Enhance)),
        ("tasks", (model, _) => model.NavigateAsync(model.Tasks)),
        ("server", (model, _) => model.NavigateAsync(model.Server)),
        ("stats", (model, _) => model.NavigateAsync(model.Stats)),
        ("nav-collapsed-groups", async (model, _) =>
        {
            await model.NavigateAsync(model.Chat);
            foreach (var group in model.Navigation.Groups.Where(group => group.Id != "text"))
            {
                group.IsExpanded = false;
            }
        }),
        ("models-discover-verified", async (model, window) => await ShowDiscoverAsync(model, window, discover => discover.ShowVerifiedCommand.Execute(null))),
        ("models-discover-hub", async (model, window) => await ShowDiscoverAsync(model, window, discover => discover.ShowHubSearchCommand.Execute(null))),
        ("models-discover-feed", async (model, window) =>
        {
            await ShowDiscoverAsync(model, window, discover => discover.ShowFeedCommand.Execute(null));
            await model.Models.Discover.Feed.LoadAsync();
            await HeadlessApp.WaitForAsync(window, () => model.Models.Discover.Feed.Entries.Count > 0);
        }),
        ("models-library", async (model, _) =>
        {
            await model.NavigateAsync(model.Models);
            model.Models.SelectedTab = ModelsPageViewModel.LibraryTab;
        }),
        ("models-library-cleanup", async (model, window) =>
        {
            await model.NavigateAsync(model.Models);
            model.Models.SelectedTab = ModelsPageViewModel.LibraryTab;
            var library = model.Models.Library;
            await library.RefreshAsync();
            library.IsSelecting = true;
            library.SelectAllCommand.Execute(null);
            foreach (var entry in library.Entries.Take(2))
            {
                entry.FeedUpdateVersion = "1.1.0";
            }

            if (library.Entries.FirstOrDefault() is { } first)
            {
                first.ShowBreakdown(new Services.Maintenance.ModelSizeBreakdown(5_000_000_000, 4_600_000_000, 12_000_000, [new("vulkan", "NVIDIA RTX PRO 4500", "spirv, pipeline-cache", 310_000_000, 640), new("cuda", "sm_120", "ptx, cubin", 60_000_000, 120)], 8_000_000, true));
                first.IsBreakdownOpen = true;
            }

            library.Confirmation = new ConfirmationViewModel("Delete 3 models? This frees 12.4 GB. 1 model is skipped because it is in use: qwen3. Unload first.", "Delete", true, () => Task.CompletedTask, _ => { });
            await HeadlessApp.RenderAsync(window);
        }),
        ("models-library-importing", async (model, window) =>
        {
            await model.NavigateAsync(model.Models);
            model.Models.SelectedTab = ModelsPageViewModel.LibraryTab;
            model.Models.Library.Confirmation = new ConfirmationViewModel("Import 2 packages (3.1 GB) into the model store? They were checked and are valid: a, b. Not imported: broken.slm: Not a valid package.", "Import", false, () => Task.CompletedTask, _ => { });
            await HeadlessApp.RenderAsync(window);
        }),
        ("models-storage", async (model, window) =>
        {
            await model.NavigateAsync(model.Models);
            model.Models.SelectedTab = ModelsPageViewModel.StorageTab;
            await HeadlessApp.WaitForAsync(window, () => model.Models.Storage.IsMeasured);
        }),
        ("models-resident", async (model, _) =>
        {
            await model.NavigateAsync(model.Models);
            model.Models.SelectedTab = ModelsPageViewModel.ResidentTab;
        }),
        ("models-downloads", async (model, _) =>
        {
            await model.NavigateAsync(model.Models);
            model.Models.SelectedTab = ModelsPageViewModel.DownloadsTab;
        }),
        ("settings-general", (model, _) => ShowSettingsAsync(model, SettingsTab.General)),
        ("settings-storage", (model, _) => ShowSettingsAsync(model, SettingsTab.Storage)),
        ("settings-models", (model, _) => ShowSettingsAsync(model, SettingsTab.Models)),
        ("settings-engine", (model, _) => ShowSettingsAsync(model, SettingsTab.Engine)),
        ("settings-updates", (model, _) => ShowSettingsAsync(model, SettingsTab.Updates)),
        ("settings-data", (model, _) => ShowSettingsAsync(model, SettingsTab.DataAndAbout)),
    ];

    public static TheoryData<string, string, int> Cases
    {
        get
        {
            var data = new TheoryData<string, string, int>();
            foreach (var (name, _) in s_pages)
            {
                foreach (string theme in new[] { "light", "dark" })
                {
                    foreach (var (width, _) in s_sizes)
                    {
                        data.Add(name, theme, (int)width);
                    }
                }
            }

            return data;
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public Task Page_RendersWithoutBindingErrorsOrClippedControls(string page, string theme, int width) => HeadlessApp.RunAsync(async () =>
    {
        double height = s_sizes.Single(size => (int)size.Width == width).Height;
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var rig = new FeedCatalogTests.Rig();
        await FeedCatalogTests.PublishAlphaAsync(rig);
        await WelcomeFeed.PublishAsync(rig);
        await DemoData.FillAsync(rig.Services);
        if (page == "chat-get-model")
        {
            foreach (var resident in rig.Services.Runtime.LoadedModels.ToList())
            {
                await rig.Services.Runtime.UnloadAsync(resident.Id);
            }
        }

        new ThemeService(Application.Current!).Apply(theme == "dark" ? AppTheme.Dark : AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(rig.All);
        var window = new MainWindow { DataContext = viewModel, Width = width, Height = height };
        window.Show();
        try
        {
            await s_pages.Single(entry => entry.Name == page).Show(viewModel, window);
            await HeadlessApp.RenderAsync(window);
            await HeadlessApp.RenderAsync(window);

            var problems = LayoutChecker.Find(window, IsChecked).Distinct().Take(8).ToList();
            Save(window, $"{page}-{theme}-{width}");
            Assert.True(problems.Count == 0, $"{page} ({theme}, {width} px):\n" + string.Join("\n", problems));
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    public static TheoryData<string, int, string> EngineCudaCases
    {
        get
        {
            var data = new TheoryData<string, int, string>();
            foreach (string variant in new[] { "cuda-chosen", "cuda-fallback", "cuda-unavailable" })
            {
                foreach (string theme in new[] { "light", "dark" })
                {
                    foreach (var (width, _) in s_sizes)
                    {
                        data.Add(theme, (int)width, variant);
                    }
                }
            }

            return data;
        }
    }

    /// <summary>
    /// Settings &gt; Engine on an NVIDIA GPU that Vulkan and CUDA both drive: the provider picker with CUDA chosen, the same with a driver too old for
    /// CUDA (the note and Vulkan in use), and a machine where CUDA is not available at all (no picker, the note).
    /// </summary>
    [Theory]
    [MemberData(nameof(EngineCudaCases))]
    public Task EngineSettings_WithCudaDevices_RendersWithoutBindingErrorsOrClippedControls(string theme, int width, string variant) => HeadlessApp.RunAsync(async () =>
    {
        double height = s_sizes.Single(size => (int)size.Width == width).Height;
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var rig = new FeedCatalogTests.Rig();
        var discovery = variant switch
        {
            "cuda-fallback" => ProviderFixtures.Discover(cudaDriver: "12.8"),
            "cuda-unavailable" => ProviderFixtures.Discover(cudaAvailability: ProviderAvailability.Unavailable("libcuda.so.1 was not found: no NVIDIA driver is installed"), cudaListed: false),
            _ => ProviderFixtures.Discover(),
        };
        ProviderFixtures.Wire(rig.Services, discovery);
        if (variant != "cuda-unavailable")
        {
            string key = rig.Services.Runtime.Gpus[0].Key;
            rig.Services.Settings.Update(current => current with { DeviceProviders = new Dictionary<string, ProviderChoice> { [key] = ProviderChoice.Cuda } });
        }

        new ThemeService(Application.Current!).Apply(theme == "dark" ? AppTheme.Dark : AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(rig.All);
        var window = new MainWindow { DataContext = viewModel, Width = width, Height = height };
        window.Show();
        try
        {
            await ShowSettingsAsync(viewModel, SettingsTab.Engine);
            await HeadlessApp.RenderAsync(window);
            await HeadlessApp.RenderAsync(window);

            var texts = window.GetVisualDescendants().OfType<Avalonia.Controls.TextBlock>().Where(block => block.IsEffectivelyVisible).Select(block => block.Text ?? "").ToList();
            Assert.Contains("Compute devices", texts);
            Assert.Contains(texts, text => text.StartsWith("Runs on ", StringComparison.Ordinal));
            Assert.Equal(variant != "cuda-chosen", texts.Any(text => text.Contains("Vulkan is used", StringComparison.Ordinal)));
            Assert.Equal(variant == "cuda-chosen", texts.Contains("Runs on CUDA"));
            var problems = LayoutChecker.Find(window, IsChecked).Distinct().Take(8).ToList();
            Save(window, $"settings-engine-{variant}-{theme}-{width}");
            Assert.True(problems.Count == 0, $"settings-engine {variant} ({theme}, {width} px):\n" + string.Join("\n", problems));
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    public static TheoryData<string, int> SearchCases
    {
        get
        {
            var data = new TheoryData<string, int>();
            foreach (string theme in new[] { "light", "dark" })
            {
                foreach (var (width, _) in s_sizes)
                {
                    data.Add(theme, (int)width);
                }
            }

            return data;
        }
    }

    /// <summary>The Search page after a hybrid Ask: highlighted passages with their places, the answer with its citations, the watch switch and the chat model picker.</summary>
    [Theory]
    [MemberData(nameof(SearchCases))]
    public Task SearchPage_WithResultsAndAnAnswer_RendersWithoutBindingErrorsOrClippedControls(string theme, int width) => HeadlessApp.RunAsync(async () =>
    {
        double height = s_sizes.Single(size => (int)size.Width == width).Height;
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var rig = new FeedCatalogTests.Rig();
        await DemoData.FillAsync(rig.Services);
        rig.Services.Runtime.Add(new Fakes.FakeModel("fake-embed") { Kind = ModelKindCatalog.Embeddings, Embedder = new Fakes.FakeTextEmbedder() });
        var chat = rig.Services.Runtime.LoadedModels.First(model => model.Id == "qwen3-0.6b");
        string folder = Path.Combine(rig.Services.Directory, "docs");
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "harbour.md"), "# Harbour\nThe ship engine roared as the ship left the harbour. The cat watched the ship from the quay.\n");
        DocumentFixtures.Pdf(Path.Combine(folder, "manual.pdf"), "Page one is about taxes.", "Engine maintenance: check the ship engine every week.");
        File.WriteAllText(Path.Combine(folder, "pets.txt"), "The dog barks. The cat sleeps.\n");

        new ThemeService(Application.Current!).Apply(theme == "dark" ? AppTheme.Dark : AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(rig.All with { Dispatcher = new AvaloniaDispatcher() });
        var window = new MainWindow { DataContext = viewModel, Width = width, Height = height };
        window.Show();
        try
        {
            await viewModel.NavigateAsync(viewModel.Search);
            var page = viewModel.Search;
            await HeadlessApp.WaitForAsync(window, () => page.HasModel && page.CanAskModel);
            page.SetFolder(folder);
            page.WatchFolder = true;
            page.IndexCommand.Execute(null);
            await page.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromSeconds(30));
            await HeadlessApp.WaitForAsync(window, () => page.HasIndex);
            page.Panel.Apply(new Dictionary<string, string> { ["mode"] = "hybrid", ["topk"] = "3" });
            page.Query = "how do I look after the ship engine";
            await HeadlessApp.WaitForAsync(window, () => page.AskCommand.CanExecute(null));
            ((Fakes.FakeModel)chat).Script.Clear();
            ((Fakes.FakeModel)chat).Script.AddRange([Services.Runtime.ChatStreamUpdate.Content("Check the ship engine every week [1], and watch it when the ship leaves the harbour [2]."),
                Services.Runtime.ChatStreamUpdate.Finished(new MessageStats())]);
            page.AskCommand.Execute(null);
            await page.Queue.Items.Last().Completion.WaitAsync(TimeSpan.FromSeconds(30));
            await HeadlessApp.WaitForAsync(window, () => page.HasResults && page.HasAnswer);
            Assert.NotEmpty(page.AnswerCitations);
            page.FocusResultCommand.Execute(page.Results[0]);
            await HeadlessApp.RenderAsync(window);
            await HeadlessApp.RenderAsync(window);

            var problems = LayoutChecker.Find(window, IsChecked).Distinct().Take(8).ToList();
            Save(window, $"search-results-{theme}-{width}");
            Assert.True(problems.Count == 0, $"search results ({theme}, {width} px):\n" + string.Join("\n", problems));
            page.WatchFolder = false;
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    private static readonly string[] s_audioModelStates = ["audio-music-ace-step", "speak-vibevoice-script", "speak-csm-script", "speak-omnivoice-design", "transcribe-speakers", "speak-license-notice"];

    public static TheoryData<string, string, int> AudioModelCases
    {
        get
        {
            var data = new TheoryData<string, string, int>();
            foreach (string state in s_audioModelStates)
            {
                foreach (string theme in new[] { "light", "dark" })
                {
                    foreach (var (width, _) in s_sizes)
                    {
                        data.Add(state, theme, (int)width);
                    }
                }
            }

            return data;
        }
    }

    /// <summary>The pages of the audio models of 2026-11-19: ACE-Step music with lyrics, VibeVoice and CSM scripts with a voice per speaker, OmniVoice languages and voice design, diarized transcripts and the licence notice.</summary>
    [Theory]
    [MemberData(nameof(AudioModelCases))]
    public Task AudioModelPage_RendersWithoutBindingErrorsOrClippedControls(string state, string theme, int width) => HeadlessApp.RunAsync(async () =>
    {
        double height = s_sizes.Single(size => (int)size.Width == width).Height;
        var sink = new BindingErrorSink();
        Logger.Sink = sink;
        using var rig = new FeedCatalogTests.Rig();
        await DemoData.FillAsync(rig.Services);
        new ThemeService(Application.Current!).Apply(theme == "dark" ? AppTheme.Dark : AppTheme.Light, "Violet");
        var viewModel = new MainWindowViewModel(rig.All with { Dispatcher = new AvaloniaDispatcher() });
        var window = new MainWindow { DataContext = viewModel, Width = width, Height = height };
        window.Show();
        try
        {
            string wav = rig.Services.Directory + "/talk.wav";
            Sharplight.Audio.WavFile.Write(wav, new Sharplight.Audio.AudioBuffer(new float[16000 * 8], 16000));
            switch (state)
            {
                case "audio-music-ace-step":
                {
                    var generator = new Fakes.FakeAudioGenerator();
                    var model = new Fakes.FakeModel("ace-step-1.5-turbo") { Kind = ModelKinds.AudioGeneration };
                    model.Clients[typeof(Sharplight.Audio.IAudioGenerator)] = generator;
                    rig.Services.Runtime.Add(model);
                    viewModel.OpenKind(ModelKinds.AudioGeneration, model);
                    await HeadlessApp.RenderAsync(window);
                    viewModel.Audio.Prompt = "upbeat synth pop, 120 bpm";
                    viewModel.Audio.Lyrics = "[verse]\nWalking down the neon street\n[chorus]\nWe are alive tonight";
                    viewModel.Audio.Panel.Apply(new Dictionary<string, string> { ["musiclanguage"] = "en", ["musicsteps"] = "8", ["shift"] = "3" });
                    viewModel.Audio.RunCommand.Execute(null);
                    await viewModel.Audio.Queue.Items.Single().Completion;
                    await HeadlessApp.WaitForAsync(window, () => viewModel.Audio.HasResult);
                    break;
                }

                case "speak-vibevoice-script":
                case "speak-csm-script":
                {
                    bool csm = state == "speak-csm-script";
                    var synthesizer = new Fakes.FakeSynthesizer(csm ? "csm" : "vibevoice") { Voices = csm ? ["0", "1"] : [] };
                    var model = new Fakes.FakeModel(csm ? "csm-1b" : "vibevoice-1.5b") { Kind = ModelKinds.SpeechSynthesis };
                    model.Clients[typeof(Sharplight.Speech.ISpeechSynthesizer)] = synthesizer;
                    rig.Services.Runtime.Add(model);
                    viewModel.OpenKind(ModelKinds.SpeechSynthesis, model);
                    await HeadlessApp.WaitForAsync(window, () => viewModel.Speak.IsMultiSpeaker);
                    viewModel.Speak.InputText = csm ? "Speaker 0: Did you hear that?\nSpeaker 1: Hear what?" : "Speaker 1: Welcome to the show.\nSpeaker 2: Thanks for having me.\nSpeaker 1: Let us begin.";
                    break;
                }

                case "speak-omnivoice-design":
                case "speak-license-notice":
                {
                    var synthesizer = new Fakes.FakeSynthesizer("omnivoice") { LanguageNames = ["abkhazian", "english", "german", "japanese"] };
                    var model = new Fakes.FakeModel("omnivoice") { Kind = ModelKinds.SpeechSynthesis };
                    model.Clients[typeof(Sharplight.Speech.ISpeechSynthesizer)] = synthesizer;
                    rig.Services.Runtime.Add(model);
                    rig.Services.Library.Entries.Add(new Services.Library.LibraryEntry
                    {
                        Id = "omnivoice",
                        DisplayName = "k2-fsa/OmniVoice",
                        Path = model.Path,
                        Kind = ModelKinds.SpeechSynthesis,
                        Manifest = new Sharplight.Hub.ModelManifest { Repository = "k2-fsa/OmniVoice", Tag = "latest", Alias = "omnivoice", Commit = "abc", Catalog = "omnivoice/safetensors" },
                    });
                    rig.Services.Hub.Catalog = new Sharplight.Hub.Catalog.VerifiedCatalog
                    {
                        Updated = "2026-11-19",
                        Entries = [new Sharplight.Hub.Catalog.CatalogEntry
                        {
                            Id = "omnivoice",
                            Name = "OmniVoice",
                            Kind = ModelKinds.SpeechSynthesis,
                            License = "other",
                            Notes = "LICENSE TERMS, read before use: the weights are CC-BY-NC (non-commercial use only); the bundled audio tokenizer is under the Boson Higgs Audio 2 Community License.",
                        }],
                    };
                    viewModel.OpenKind(ModelKinds.SpeechSynthesis, model);
                    await HeadlessApp.WaitForAsync(window, () => viewModel.Speak.HasLicenseNotice && viewModel.Speak.Panel.Parameters.First(parameter => parameter.Key == "language").IsVisibleNow);
                    viewModel.Speak.InputText = "Guten Tag, wie geht es Ihnen?";
                    viewModel.Speak.Panel.Apply(new Dictionary<string, string> { ["language"] = "german", ["design"] = "female, young adult, british accent" });
                    break;
                }

                case "transcribe-speakers":
                {
                    var recognizer = new Fakes.FakeRecognizer();
                    recognizer.Script.Clear();
                    recognizer.Script.AddRange(
                    [
                        new() { Start = TimeSpan.Zero, End = TimeSpan.FromSeconds(2), Text = " Hello, how are you?", Speaker = "0" },
                        new() { Start = TimeSpan.FromSeconds(2), End = TimeSpan.FromSeconds(4), Text = " Fine, thanks. And you?", Speaker = "1" },
                        new() { Start = TimeSpan.FromSeconds(4), End = TimeSpan.FromSeconds(7), Text = " Great, let us get started.", Speaker = "0" },
                    ]);
                    var model = new Fakes.FakeModel("vibevoice-asr") { Kind = ModelKinds.SpeechRecognition };
                    model.Clients[typeof(Sharplight.Speech.ISpeechRecognizer)] = recognizer;
                    rig.Services.Runtime.Add(model);
                    viewModel.OpenKind(ModelKinds.SpeechRecognition, model);
                    await viewModel.Transcribe.SetAudioAsync(wav);
                    viewModel.Transcribe.RunCommand.Execute(null);
                    await viewModel.Transcribe.Queue.Items.Single().Completion;
                    await HeadlessApp.WaitForAsync(window, () => viewModel.Transcribe.Segments.Count == 3 && viewModel.Transcribe.HasSpeakers);
                    break;
                }
            }

            await HeadlessApp.RenderAsync(window);
            await HeadlessApp.RenderAsync(window);

            var problems = LayoutChecker.Find(window, IsChecked).Distinct().Take(8).ToList();
            Save(window, $"{state}-{theme}-{width}");
            Assert.True(problems.Count == 0, $"{state} ({theme}, {width} px):\n" + string.Join("\n", problems));
        }
        finally
        {
            window.Close();
            Dispatcher.UIThread.RunJobs();
            Logger.Sink = null;
        }

        Assert.True(sink.Messages.Count == 0, string.Join("\n", sink.Messages.Distinct()));
    });

    /// <summary>Everything except sliders (their template overlaps the row's caption by design) and bare text (its line boxes overlap by a pixel or two).</summary>
    private static bool IsChecked(Avalonia.Visual visual) => visual is not (Avalonia.Controls.Slider or Avalonia.Controls.TextBlock);

    private static async Task ShowDiscoverAsync(MainWindowViewModel model, MainWindow window, Action<HubSearchViewModel> choose)
    {
        await model.NavigateAsync(model.Models);
        model.Models.SelectedTab = ModelsPageViewModel.DiscoverTab;
        await HeadlessApp.WaitForAsync(window, () => !model.Models.Discover.IsLoadingHighlights);
        choose(model.Models.Discover);
    }

    private static async Task ShowSettingsAsync(MainWindowViewModel model, SettingsTab tab)
    {
        await model.NavigateAsync(model.Settings);
        model.Settings.ShowTab(tab);
    }

    private static void Save(MainWindow window, string name)
    {
        string directory = Environment.GetEnvironmentVariable("SHARPLIGHT_STUDIO_SCREENSHOTS") is { Length: > 0 } configured
            ? configured
            : Path.Combine(AppContext.BaseDirectory, "page-screenshots");
        Directory.CreateDirectory(directory);
        using var frame = window.CaptureRenderedFrame();
        Assert.NotNull(frame);
        frame.Save(Path.Combine(directory, $"studio-page-{name}.png"), new PngBitmapEncoderOptions());
    }
}
