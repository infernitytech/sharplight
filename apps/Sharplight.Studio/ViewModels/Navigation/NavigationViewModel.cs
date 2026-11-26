using System.ComponentModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Models;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Settings;
using Sharplight.Studio.ViewModels.Inference;
using Sharplight.Studio.ViewModels.Models;

namespace Sharplight.Studio.ViewModels.Navigation;

/// <summary>
/// The navigation of the main window: root groups by the modality of what a page produces (Text, Images, Video, Audio) with the task
/// pages below them, a Models group (the tabs of the Models page and the Tasks overview), and the app-level entries at the bottom
/// (Server, Statistics, Settings). It is also the routing table from a model kind to its page (<see cref="ForKind"/>): the kinds come
/// from the pages themselves, so the routing knows kinds and nothing about particular models. Which groups are collapsed and whether
/// the navigation is folded to a rail are kept in the settings.
/// </summary>
internal sealed partial class NavigationViewModel : ObservableObject
{
    /// <summary>Below this window width (device-independent pixels) the navigation is a rail of group icons.</summary>
    public const double RailWidth = 1100;

    private readonly MainWindowViewModel _main;
    private readonly ISettingsService _settings;
    private bool _restoring;

    /// <param name="main">The main window, whose pages the entries show.</param>
    /// <param name="settings">Where the collapsed groups and the rail choice are kept.</param>
    public NavigationViewModel(MainWindowViewModel main, ISettingsService settings)
    {
        _main = main;
        _settings = settings;
        Groups =
        [
            new("text", "Text", "IconDocument",
            [
                Item("Chat", "Chat with language models, also about images and documents", main.Chat, [ModelKinds.TextGeneration]),
                Item("Search", "Search and ask a folder of documents (embeddings and rerankers)", main.Search,
                    [ModelKindCatalog.Reranker, ModelKinds.TextEmbedding, ModelKinds.TextReranking]),
            ]),
            new("images", "Images", "IconImage",
            [
                Item("Generate", "Generate and edit images from a prompt", main.Image),
                Item("Vision", "Classify, detect, segment, estimate depth, read text (OCR) and upscale images", main.Vision),
            ]),
            new("video", "Video", "IconVideo",
            [
                Item("Generate", "Generate videos from a prompt or an image", main.Video),
                Item("Enhance", "Restore and enlarge videos and images", main.Enhance),
            ]),
            new("audio", "Audio", "IconWave",
            [
                Item("Transcribe", "Speech to text", main.Transcribe),
                Item("Speak", "Text to speech", main.Speak),
                Item("Music & tools", "Generate music, separate sources, tag sounds, audio codecs and voice activity", main.Audio),
            ]),
            new("models", "Models", "IconModels",
            [
                Tab("Library", "The installed models", "IconModels", ModelsPageViewModel.LibraryTab),
                Tab("Discover", "Find models to download", "IconGlobe", ModelsPageViewModel.DiscoverTab),
                Tab("Downloads", "Running and finished downloads", "IconDownload", ModelsPageViewModel.DownloadsTab),
                Tab("Loaded", "The models in memory", "IconMemory", ModelsPageViewModel.ResidentTab),
                Tab("Storage", "Disk use and cleanup", "IconFolder", ModelsPageViewModel.StorageTab),
                Item("Tasks", "Every model kind and where it stands", main.Tasks),
            ]),
        ];
        Footer =
        [
            Item("Server", "The OpenAI- and Anthropic-compatible API", main.Server),
            Item("Statistics", "Throughput and memory over time", main.Stats),
            Item("Settings", "Appearance, storage, engine and updates", main.Settings),
        ];
        Items = [.. Groups.SelectMany(group => group.Items), .. Footer];

        Restore(settings.Current);
        foreach (var group in Groups)
        {
            group.PropertyChanged += OnGroupChanged;
        }

        main.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(MainWindowViewModel.CurrentPage))
            {
                Sync();
            }
        };
        main.Models.PropertyChanged += (_, args) =>
        {
            if (args.PropertyName == nameof(ModelsPageViewModel.SelectedTab))
            {
                Sync();
            }
        };
        Sync();
    }

    /// <summary>The collapsible groups, top to bottom.</summary>
    public IReadOnlyList<NavigationGroupViewModel> Groups { get; }

    /// <summary>The app-level entries pinned to the bottom.</summary>
    public IReadOnlyList<NavigationItemViewModel> Footer { get; }

    /// <summary>Every entry: the groups' in order, then the footer's.</summary>
    public IReadOnlyList<NavigationItemViewModel> Items { get; }

    /// <summary>Raised after an entry was opened (the rail closes its flyout).</summary>
    public event EventHandler? ItemOpened;

    /// <summary>The entry of the visible page, or null (the welcome flow has none).</summary>
    [ObservableProperty]
    public partial NavigationItemViewModel? Current { get; private set; }

    /// <summary>The user folded the navigation to a rail (kept in the settings).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsRail), nameof(IsExpandedSidebar), nameof(CanToggleCompact))]
    public partial bool IsCompact { get; set; }

    /// <summary>The window is narrower than <see cref="RailWidth"/> (set by the view).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsRail), nameof(IsExpandedSidebar), nameof(CanToggleCompact))]
    public partial bool IsNarrow { get; set; }

    /// <summary>The navigation shows only the group icons, each opening its entries in a flyout.</summary>
    public bool IsRail => IsCompact || IsNarrow;

    /// <summary>The navigation shows the groups with their labels and entries.</summary>
    public bool IsExpandedSidebar => !IsRail;

    /// <summary>The fold button is shown: a narrow window is always a rail.</summary>
    public bool CanToggleCompact => !IsNarrow;

    /// <summary>The entry whose page runs models of <paramref name="kind"/>, or null when no page does.</summary>
    /// <param name="kind">The kind (<see cref="ModelKinds"/>).</param>
    public NavigationItemViewModel? ForKind(string kind)
        => Items.FirstOrDefault(item => item.Kinds.Contains(kind, StringComparer.OrdinalIgnoreCase));

    /// <summary>The group <paramref name="id"/>, or null.</summary>
    /// <param name="id">The group's id.</param>
    public NavigationGroupViewModel? Group(string id) => Groups.FirstOrDefault(group => group.Id == id);

    /// <summary>Folds the navigation to a rail, or unfolds it.</summary>
    [RelayCommand]
    private void ToggleCompact() => IsCompact = !IsCompact;

    /// <summary>Expands or collapses a group.</summary>
    [RelayCommand]
    private static void ToggleGroup(NavigationGroupViewModel? group) => group?.Toggle();

    partial void OnIsCompactChanged(bool value)
    {
        if (!_restoring)
        {
            Save(settings => settings with { CompactNavigation = value });
        }
    }

    private NavigationItemViewModel Item(string title, string description, PageViewModel page, IReadOnlyList<string>? extraKinds = null)
    {
        IReadOnlyList<string> kinds = [.. (page as InferencePageViewModel)?.Kinds ?? [], .. extraKinds ?? []];
        return new NavigationItemViewModel(title, description, page.Icon, page, null, kinds, showsBadge: true, Open);
    }

    /// <summary>An entry for a tab of the Models page; the page's badge (downloads, model updates) is on the Library entry.</summary>
    private NavigationItemViewModel Tab(string title, string description, string icon, int tab)
        => new(title, description, icon, _main.Models, tab, [], showsBadge: tab == ModelsPageViewModel.LibraryTab, Open);

    private void Open(NavigationItemViewModel item)
    {
        if (item.Tab is { } tab)
        {
            _main.Models.SelectedTab = tab;
        }

        _ = _main.NavigateAsync(item.Page);
        Sync();
        ItemOpened?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>Marks the entry of the visible page (and tab) and expands its group.</summary>
    private void Sync()
    {
        var page = _main.CurrentPage;
        var candidates = Items.Where(item => ReferenceEquals(item.Page, page)).ToList();
        var current = candidates.FirstOrDefault(item => item.Tab == _main.Models.SelectedTab)
            ?? candidates.FirstOrDefault(item => item.Tab is null)
            ?? candidates.FirstOrDefault();
        foreach (var item in Items)
        {
            item.IsCurrent = ReferenceEquals(item, current);
        }

        foreach (var group in Groups)
        {
            group.ContainsCurrent = current?.Group == group;
        }

        if (current?.Group is { IsExpanded: false } parent)
        {
            parent.IsExpanded = true;
        }

        Current = current;
    }

    private void Restore(AppSettings settings)
    {
        _restoring = true;
        try
        {
            IsCompact = settings.CompactNavigation;
            foreach (var group in Groups)
            {
                group.IsExpanded = !settings.CollapsedNavigationGroups.Contains(group.Id);
            }
        }
        finally
        {
            _restoring = false;
        }
    }

    private void OnGroupChanged(object? sender, PropertyChangedEventArgs args)
    {
        if (args.PropertyName != nameof(NavigationGroupViewModel.IsExpanded) || _restoring)
        {
            return;
        }

        string[] collapsed = [.. Groups.Where(group => !group.IsExpanded).Select(group => group.Id)];
        Save(settings => settings with { CollapsedNavigationGroups = collapsed });
    }

    /// <summary>Remembers a navigation preference; one that cannot be written (a read-only or full disk) is kept for this session only.</summary>
    private void Save(Func<AppSettings, AppSettings> change)
    {
        try
        {
            _settings.Update(change);
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        {
            // The navigation keeps working with the state in memory.
        }
    }
}
