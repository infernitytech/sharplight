using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Chat;

/// <summary>The chat page: the conversation list, the model picker and the open conversation.</summary>
internal sealed partial class ChatPageViewModel : PageViewModel
{
    private readonly StudioServices _services;
    private readonly INavigator _navigator;
    private const int MaxOpenConversations = 8;

    private readonly List<ConversationItemViewModel> _all = [];
    private readonly Dictionary<string, ConversationViewModel> _open = new(StringComparer.Ordinal);
    private bool _loaded;

    public ChatPageViewModel(StudioServices services, INavigator navigator)
    {
        _services = services;
        _navigator = navigator;
        services.Runtime.ModelsChanged += (_, _) => services.Dispatcher.Post(RefreshModels);
        RefreshModels();
    }

    /// <inheritdoc/>
    public override string Title => "Chat";

    /// <inheritdoc/>
    public override string Icon => "IconChat";

    /// <summary>Conversations matching the filter, most recent first.</summary>
    public ObservableCollection<ConversationItemViewModel> Conversations { get; } = [];

    /// <summary>Loaded models.</summary>
    public ObservableCollection<ModelOptionViewModel> Models { get; } = [];

    /// <summary>Filter text of the conversation list.</summary>
    [ObservableProperty]
    public partial string FilterText { get; set; } = "";

    /// <summary>The conversation selected in the list.</summary>
    [ObservableProperty]
    public partial ConversationItemViewModel? SelectedConversation { get; set; }

    /// <summary>The open conversation.</summary>
    [ObservableProperty]
    public partial ConversationViewModel? Current { get; set; }

    /// <summary>The model answers come from.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SelectedBadges))]
    public partial ModelOptionViewModel? SelectedModel { get; set; }

    /// <summary>The settings panel is open.</summary>
    [ObservableProperty]
    public partial bool IsSettingsOpen { get; set; } = true;

    /// <summary>At least one model is loaded.</summary>
    public bool HasModels => Models.Count > 0;

    /// <summary>Badges of the selected model.</summary>
    public CapabilityBadges? SelectedBadges => SelectedModel?.Badges;

    /// <inheritdoc/>
    public override async Task ActivateAsync()
    {
        if (_loaded)
        {
            return;
        }

        _loaded = true;
        var summaries = await _services.Conversations.ListAsync();
        _all.Clear();
        _all.AddRange(summaries.Select(summary => new ConversationItemViewModel(summary)));
        ApplyFilter();
        if (Conversations.Count > 0)
        {
            SelectedConversation = Conversations[0];
        }
        else
        {
            await NewChatAsync();
        }
    }

    /// <summary>Selects <paramref name="model"/> (or keeps the selection).</summary>
    public void SelectModel(ILoadedModel? model)
    {
        RefreshModels();
        if (model is not null && Models.FirstOrDefault(option => option.Id == model.Id) is { } option)
        {
            SelectedModel = option;
        }
    }

    partial void OnFilterTextChanged(string value) => ApplyFilter();

    partial void OnSelectedConversationChanged(ConversationItemViewModel? value)
    {
        if (value is not null && value.Id != Current?.Id)
        {
            _ = OpenAsync(value.Id);
        }
    }

    partial void OnSelectedModelChanged(ModelOptionViewModel? value)
    {
        if (Current is null)
        {
            return;
        }

        if (value is not null)
        {
            Current.ModelId = value.Id;
        }

        Current.OnModelChanged();
    }

    [RelayCommand]
    private async Task NewChatAsync()
    {
        var settings = _services.Settings.Current;
        var conversation = new Conversation
        {
            SystemPrompt = settings.DefaultSystemPrompt,
            Sampling = settings.DefaultSampling,
            EnableThinking = settings.EnableThinking,
            ModelId = SelectedModel?.Id,
        };
        await _services.Conversations.SaveAsync(conversation);
        var item = new ConversationItemViewModel(new ConversationSummary(conversation.Id, conversation.Title, conversation.UpdatedAt, 0));
        _all.Insert(0, item);
        FilterText = "";
        ApplyFilter();
        Open(conversation);
        SelectedConversation = item;
    }

    [RelayCommand]
    private async Task DeleteConversationAsync(ConversationItemViewModel item)
    {
        if (_open.Remove(item.Id, out var open))
        {
            open.CancelGeneration();
        }

        await _services.Conversations.DeleteAsync(item.Id);
        _all.Remove(item);
        Conversations.Remove(item);
        if (Current?.Id == item.Id)
        {
            Current = null;
            if (Conversations.Count > 0)
            {
                SelectedConversation = Conversations[0];
            }
            else
            {
                await NewChatAsync();
            }
        }
    }

    [RelayCommand]
    private static void Rename(ConversationItemViewModel item) => item.IsRenaming = true;

    [RelayCommand]
    private async Task CommitRenameAsync(ConversationItemViewModel item)
    {
        item.IsRenaming = false;
        var conversation = await _services.Conversations.LoadAsync(item.Id);
        if (conversation is null)
        {
            return;
        }

        var title = item.Title.Trim().Length == 0 ? "New chat" : item.Title.Trim();
        await _services.Conversations.SaveAsync(conversation with { Title = title });
        if (_open.TryGetValue(item.Id, out var open))
        {
            open.SetTitle(title);
        }
    }

    [RelayCommand]
    private void ToggleSettings() => IsSettingsOpen = !IsSettingsOpen;

    [RelayCommand]
    private void OpenLibrary() => _navigator.OpenLibrary();

    [RelayCommand]
    private void OpenDiscover() => _navigator.OpenDiscover();


    private async Task OpenAsync(string id)
    {
        var conversation = await _services.Conversations.LoadAsync(id);
        if (conversation is not null)
        {
            Open(conversation);
        }
    }

    private void Open(Conversation conversation)
    {
        // Conversations stay open while they generate, so answers keep streaming in the background and two chats can run
        // at once (on the same model they are batched together, on different models they run side by side).
        if (!_open.TryGetValue(conversation.Id, out var view))
        {
            ConversationViewModel? created = null;
            created = new ConversationViewModel(conversation, _services, () => ModelOf(created is null ? conversation.ModelId : created.ModelId), OnConversationChanged);
            created.PropertyChanged += (_, args) =>
            {
                if (args.PropertyName == nameof(ConversationViewModel.IsGenerating)
                    && _all.FirstOrDefault(item => item.Id == created.Id) is { } item)
                {
                    item.IsGenerating = created.IsGenerating;
                }
            };
            _open[conversation.Id] = created;
            view = created;
        }

        Current = view;
        if (view.ModelId is { } modelId && Models.FirstOrDefault(option => option.Id == modelId) is { } option)
        {
            SelectedModel = option;
        }
        else if (SelectedModel is not null)
        {
            view.ModelId = SelectedModel.Id;
        }

        view.OnModelChanged();
        TrimOpen();
    }

    /// <summary>The model a conversation uses: its own when resident, else the page's selection.</summary>
    private ILoadedModel? ModelOf(string? modelId)
    {
        var models = _services.Runtime.LoadedModels;
        return models.FirstOrDefault(model => model.Id == modelId) ?? SelectedModel?.Model;
    }

    /// <summary>Forgets idle conversations other than the current one, keeping memory bounded.</summary>
    private void TrimOpen()
    {
        foreach (var (id, view) in _open.ToList())
        {
            if (view != Current && !view.IsGenerating && _open.Count > MaxOpenConversations)
            {
                _open.Remove(id);
            }
        }
    }

    private void OnConversationChanged(Conversation conversation)
    {
        var item = _all.FirstOrDefault(candidate => candidate.Id == conversation.Id);
        if (item is null)
        {
            return;
        }

        item.Title = conversation.Title;
        item.UpdatedAt = conversation.UpdatedAt;
        if (_all.IndexOf(item) > 0)
        {
            _all.Remove(item);
            _all.Insert(0, item);
            ApplyFilter();
        }
    }

    private void ApplyFilter()
    {
        var filter = FilterText.Trim();
        var selected = SelectedConversation;
        var visible = _all.Where(item => filter.Length == 0 || item.Title.Contains(filter, StringComparison.OrdinalIgnoreCase)).ToList();
        if (visible.SequenceEqual(Conversations))
        {
            return;
        }

        Conversations.Clear();
        foreach (var item in visible)
        {
            Conversations.Add(item);
        }

        if (selected is not null && Conversations.Contains(selected))
        {
            SelectedConversation = selected;
        }
    }

    private void RefreshModels()
    {
        var loaded = _services.Runtime.LoadedModels;
        var selectedId = SelectedModel?.Id;
        Models.Clear();
        foreach (var model in loaded)
        {
            Models.Add(new ModelOptionViewModel(model));
        }

        SelectedModel = Models.FirstOrDefault(option => option.Id == selectedId) ?? Models.FirstOrDefault();
        OnPropertyChanged(nameof(HasModels));
    }
}
