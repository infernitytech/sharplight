using System.Collections.ObjectModel;
using System.Globalization;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services;
using Sharplight.Studio.Services.Documents;
using Sharplight.Studio.Services.Platform;
using Sharplight.Studio.Services.Runtime;

namespace Sharplight.Studio.ViewModels.Chat;

/// <summary>
/// An open conversation: the active branch of messages, the composer with its attachments, the per-chat settings, and
/// generation (send, stop, regenerate, edit, switch versions).
/// </summary>
internal sealed partial class ConversationViewModel : ObservableObject
{
    private static readonly string[] s_imageExtensions = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"];

    private readonly StudioServices _services;
    private readonly Func<ILoadedModel?> _currentModel;
    private readonly Action<Conversation> _changed;
    private readonly bool _loading;
    private Conversation _conversation;
    private CancellationTokenSource? _generation;

    public ConversationViewModel(Conversation conversation, StudioServices services, Func<ILoadedModel?> currentModel, Action<Conversation> changed)
    {
        _conversation = conversation;
        _services = services;
        _currentModel = currentModel;
        _changed = changed;
        _loading = true;
        SystemPrompt = conversation.SystemPrompt;
        EnableThinking = conversation.EnableThinking;
        ApplySampling(conversation.Sampling);
        Presets = [.. SamplingPreset.Defaults, .. services.Settings.Current.Presets];
        _loading = false;
        Attachments.CollectionChanged += (_, _) => OnComposerChanged();
        RebuildMessages();
        UpdateContextInfo();
    }

    /// <summary>The conversation as stored.</summary>
    public Conversation Conversation => _conversation;

    /// <summary>The conversation id.</summary>
    public string Id => _conversation.Id;

    /// <summary>The model this conversation talks to (kept per conversation; switching it mid-conversation applies to the next answer).</summary>
    public string? ModelId
    {
        get => _conversation.ModelId;
        set
        {
            if (value != _conversation.ModelId)
            {
                _conversation = _conversation with { ModelId = value };
                OnPropertyChanged();
                _ = _services.Conversations.SaveAsync(_conversation);
            }
        }
    }

    /// <summary>Replaces the title (after a rename in the list).</summary>
    public void SetTitle(string title) => _conversation = _conversation with { Title = title };

    /// <summary>The active branch (system prompt excluded).</summary>
    public ObservableCollection<MessageViewModel> Messages { get; } = [];

    /// <summary>Attachments of the message being written.</summary>
    public ObservableCollection<AttachmentViewModel> Attachments { get; } = [];

    /// <summary>Sampling presets (built-in and saved).</summary>
    [ObservableProperty]
    public partial IReadOnlyList<SamplingPreset> Presets { get; set; }

    /// <summary>The preset last applied.</summary>
    [ObservableProperty]
    public partial SamplingPreset? SelectedPreset { get; set; }

    /// <summary>The message being written.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand))]
    public partial string InputText { get; set; } = "";

    /// <summary>An answer is being generated.</summary>
    [ObservableProperty]
    [NotifyCanExecuteChangedFor(nameof(SendCommand), nameof(StopCommand))]
    public partial bool IsGenerating { get; set; }

    /// <summary>A hint or problem to show above the composer.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(HasNotice))]
    public partial string? Notice { get; set; }

    /// <summary>Instructions sent as the system message.</summary>
    [ObservableProperty]
    public partial string SystemPrompt { get; set; } = "";

    /// <summary>Let reasoning models think first.</summary>
    [ObservableProperty]
    public partial bool EnableThinking { get; set; }

    /// <summary>Temperature.</summary>
    [ObservableProperty]
    public partial double Temperature { get; set; }

    /// <summary>Top-p.</summary>
    [ObservableProperty]
    public partial double TopP { get; set; }

    /// <summary>Top-k.</summary>
    [ObservableProperty]
    public partial double TopK { get; set; }

    /// <summary>Min-p.</summary>
    [ObservableProperty]
    public partial double MinP { get; set; }

    /// <summary>Repetition penalty.</summary>
    [ObservableProperty]
    public partial double RepetitionPenalty { get; set; }

    /// <summary>Maximum answer length (0 = until the context is full).</summary>
    [ObservableProperty]
    public partial double MaxTokens { get; set; }

    /// <summary>Seed as typed (empty = random).</summary>
    [ObservableProperty]
    public partial string SeedText { get; set; } = "";

    /// <summary>Name for saving the current settings as a preset.</summary>
    [ObservableProperty]
    public partial string NewPresetName { get; set; } = "";

    /// <summary>Context use: <c>1.2k / 8k tokens</c>.</summary>
    [ObservableProperty]
    public partial string ContextText { get; set; } = "";

    /// <summary>Context use as a fraction (0..1).</summary>
    [ObservableProperty]
    public partial double ContextFraction { get; set; }

    /// <summary>True when a notice is shown.</summary>
    public bool HasNotice => !string.IsNullOrEmpty(Notice);

    /// <summary>True when the chat has no messages yet.</summary>
    public bool IsEmpty => Messages.Count == 0;

    /// <summary>The composer has attachments.</summary>
    public bool HasAttachments => Attachments.Count > 0;

    /// <summary>The composer has an image the selected model cannot read.</summary>
    public bool ImageNotSupported => Attachments.Any(attachment => attachment.IsImage) && !SelectedModelHas(ModelCapabilities.Vision);

    /// <summary>The selected model emits reasoning (shows the thinking toggle).</summary>
    public bool SupportsThinking => SelectedModelHas(ModelCapabilities.Reasoning);

    /// <summary>Applies a preset.</summary>
    partial void OnSelectedPresetChanged(SamplingPreset? value)
    {
        if (value is not null)
        {
            ApplySampling(value.Sampling);
        }
    }

    partial void OnSystemPromptChanged(string value) => SaveSettings();

    partial void OnEnableThinkingChanged(bool value) => SaveSettings();

    partial void OnTemperatureChanged(double value) => SaveSettings();

    partial void OnTopPChanged(double value) => SaveSettings();

    partial void OnTopKChanged(double value) => SaveSettings();

    partial void OnMinPChanged(double value) => SaveSettings();

    partial void OnRepetitionPenaltyChanged(double value) => SaveSettings();

    partial void OnMaxTokensChanged(double value) => SaveSettings();

    partial void OnSeedTextChanged(string value) => SaveSettings();

    /// <summary>Re-evaluates what depends on the selected model.</summary>
    public void OnModelChanged()
    {
        OnPropertyChanged(nameof(ImageNotSupported));
        OnPropertyChanged(nameof(SupportsThinking));
        SendCommand.NotifyCanExecuteChanged();
        UpdateContextInfo();
    }

    /// <summary>Attaches files: images become image attachments, documents are extracted to text.</summary>
    public async Task AddFilesAsync(IEnumerable<string> paths)
    {
        foreach (var path in paths)
        {
            try
            {
                if (s_imageExtensions.Contains(Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
                {
                    AddImage(await File.ReadAllBytesAsync(path), MediaTypeOf(path), Path.GetFileName(path));
                }
                else if (_services.Documents.CanExtract(path))
                {
                    var document = await _services.Documents.ExtractAsync(path, CancellationToken.None);
                    var record = new AttachmentRecord { Kind = AttachmentKind.Document, Name = document.Name, MediaType = document.MediaType, Text = document.Text };
                    Attachments.Add(new AttachmentViewModel(record, EstimateTokens(document.Text)));
                }
                else
                {
                    Notice = $"{Path.GetFileName(path)}: unsupported file type. Attach images, PDF, DOCX, Markdown, text or source files.";
                }
            }
            catch (Exception error) when (error is IOException or NotSupportedException or UnauthorizedAccessException or InvalidOperationException)
            {
                Notice = $"Cannot attach {Path.GetFileName(path)}: {error.Message}";
            }
        }
    }

    /// <summary>Attaches an image (from a file, the clipboard or a drop).</summary>
    public void AddImage(byte[] data, string mediaType, string name)
    {
        Attachments.Add(new AttachmentViewModel(new AttachmentRecord { Kind = AttachmentKind.Image, Name = name, MediaType = mediaType, Data = data }));
    }

    /// <summary>Cancels a running generation.</summary>
    public void CancelGeneration() => _generation?.Cancel();

    [RelayCommand]
    private async Task AttachAsync()
    {
        var filters = new List<FileFilter>
        {
            new("Images and documents", [.. s_imageExtensions.Concat(DocumentExtractor.SupportedExtensions).Select(extension => "*" + extension)]),
            new("All files", ["*"]),
        };
        await AddFilesAsync(await _services.Platform.PickFilesAsync("Attach files", filters, allowMultiple: true));
    }

    [RelayCommand]
    private void RemoveAttachment(AttachmentViewModel attachment) => Attachments.Remove(attachment);

    private bool CanSend() => !IsGenerating && (InputText.Trim().Length > 0 || Attachments.Count > 0);

    [RelayCommand(CanExecute = nameof(CanSend))]
    private async Task SendAsync()
    {
        if (_currentModel() is not { } model)
        {
            Notice = "Load a model to start chatting (Models → Library → Load).";
            return;
        }

        if (ImageNotSupported)
        {
            Notice = $"{model.Id} cannot read images: the engine has no vision tower for it. Remove the images or pick a model with the Vision badge.";
            return;
        }

        Notice = null;
        var text = InputText.Trim();
        var attachments = FitAttachments(model, text);
        var user = new ChatMessageRecord { ParentId = _conversation.CurrentLeafId, Role = ChatRole.User, Content = text, Attachments = attachments };
        _conversation = ConversationTree.Append(_conversation, user);
        if (_conversation.Messages.Count(message => message.Role == ChatRole.User) == 1)
        {
            _conversation = _conversation with { Title = ConversationTree.TitleFrom(text.Length > 0 ? text : attachments[0].Name) };
        }

        InputText = "";
        Attachments.Clear();
        RebuildMessages();
        await GenerateAsync(user.Id, model);
    }

    private bool CanStop() => IsGenerating;

    [RelayCommand(CanExecute = nameof(CanStop))]
    private void Stop() => _generation?.Cancel();

    [RelayCommand]
    private async Task RegenerateAsync(MessageViewModel message)
    {
        if (IsGenerating || message.Record.ParentId is not { } parent)
        {
            return;
        }

        if (_currentModel() is not { } model)
        {
            Notice = "Load a model to regenerate the answer.";
            return;
        }

        await GenerateAsync(parent, model);
    }

    [RelayCommand]
    private void Edit(MessageViewModel message)
    {
        message.EditText = message.Content;
        message.IsEditing = true;
    }

    [RelayCommand]
    private void CancelEdit(MessageViewModel message) => message.IsEditing = false;

    [RelayCommand]
    private async Task SaveEditAsync(MessageViewModel message)
    {
        message.IsEditing = false;
        if (IsGenerating || message.EditText.Trim() == message.Content.Trim())
        {
            return;
        }

        var edited = new ChatMessageRecord
        {
            ParentId = message.Record.ParentId,
            Role = ChatRole.User,
            Content = message.EditText.Trim(),
            Attachments = message.Record.Attachments,
        };
        _conversation = ConversationTree.Append(_conversation, edited);
        RebuildMessages();
        if (_currentModel() is { } model)
        {
            await GenerateAsync(edited.Id, model);
        }
        else
        {
            await SaveAsync();
        }
    }

    [RelayCommand]
    private Task PreviousVersionAsync(MessageViewModel message) => SwitchVersionAsync(message, -1);

    [RelayCommand]
    private Task NextVersionAsync(MessageViewModel message) => SwitchVersionAsync(message, 1);

    [RelayCommand]
    private Task CopyAsync(MessageViewModel message) => _services.Platform.CopyTextAsync(message.Content);

    [RelayCommand]
    private void UseSuggestion(string text) => InputText = text;

    [RelayCommand]
    private void SavePreset()
    {
        var name = NewPresetName.Trim();
        if (name.Length == 0)
        {
            return;
        }

        var preset = new SamplingPreset(name, CurrentSampling());
        _services.Settings.Update(settings => settings with { Presets = [.. settings.Presets.Where(existing => existing.Name != name), preset] });
        Presets = [.. SamplingPreset.Defaults, .. _services.Settings.Current.Presets];
        NewPresetName = "";
    }

    private async Task SwitchVersionAsync(MessageViewModel message, int offset)
    {
        if (IsGenerating)
        {
            return;
        }

        _conversation = ConversationTree.SwitchVersion(_conversation, message.Record, offset);
        RebuildMessages();
        UpdateContextInfo();
        await SaveAsync();
    }

    private async Task GenerateAsync(string parentId, ILoadedModel model)
    {
        var input = BuildInput(ConversationTree.PathTo(_conversation, parentId));
        var assistant = new ChatMessageRecord { ParentId = parentId, Role = ChatRole.Assistant, ModelId = model.Id };
        _conversation = ConversationTree.Append(_conversation, assistant) with { ModelId = model.Id };
        RebuildMessages();
        var view = Messages[^1];
        view.IsStreaming = true;
        IsGenerating = true;
        using var generation = new CancellationTokenSource();
        _generation = generation;
        MessageStats? stats = null;
        string? error = null;
        try
        {
            await foreach (var update in model.StreamChatAsync(input, generation.Token))
            {
                switch (update.Kind)
                {
                    case ChatStreamKind.Content:
                        view.AppendContent(update.Text);
                        break;
                    case ChatStreamKind.Reasoning:
                        view.AppendReasoning(update.Text);
                        break;
                    case ChatStreamKind.ToolCall:
                        view.AppendContent($"\n\n`→ {update.Text}`\n");
                        break;
                    case ChatStreamKind.Finished:
                        stats = update.Stats;
                        error = update.Error;
                        break;
                }
            }
        }
        catch (Exception failure) when (failure is not OutOfMemoryException)
        {
            error = failure.Message;
        }
        finally
        {
            _generation = null;
            view.IsStreaming = false;
            IsGenerating = false;
        }

        var record = assistant with
        {
            Content = view.Content,
            Reasoning = view.Reasoning.Length > 0 ? view.Reasoning : null,
            Stats = stats,
            Error = error,
        };
        _conversation = ConversationTree.Replace(_conversation, record);
        view.Update(record);
        UpdateContextInfo();
        await SaveAsync();
    }

    /// <summary>The model input for the branch ending at the last message of <paramref name="path"/>.</summary>
    internal ChatInput BuildInput(IReadOnlyList<ChatMessageRecord> path)
    {
        var messages = new List<ChatInputMessage>();
        if (SystemPrompt.Trim().Length > 0)
        {
            messages.Add(ChatInputMessage.FromText(ChatRole.System, SystemPrompt.Trim()));
        }

        foreach (var message in path)
        {
            if (message.Role == ChatRole.Assistant && (message.Error is not null || message.Content.Length == 0))
            {
                continue;
            }

            var images = message.Attachments.Where(attachment => attachment.Kind == AttachmentKind.Image && attachment.Data is not null)
                .Select(attachment => new ImageContent(attachment.Data!, attachment.MediaType)).ToList();
            messages.Add(new ChatInputMessage(message.Role, PromptComposer.Compose(message.Content, message.Attachments), images));
        }

        return new ChatInput(messages, CurrentSampling(), EnableThinking);
    }

    /// <summary>
    /// The composer's attachments as stored: documents fitted into what is left of the context (shared equally, smaller
    /// documents leave their unused share to the others), images unchanged.
    /// </summary>
    private List<AttachmentRecord> FitAttachments(ILoadedModel model, string text)
    {
        var records = Attachments.Select(attachment => attachment.Record).ToList();
        var documents = records.Where(record => record.Kind == AttachmentKind.Document).OrderBy(record => record.Text?.Length ?? 0).ToList();
        if (documents.Count == 0)
        {
            return records;
        }

        var history = BuildInput(ConversationTree.ActivePath(_conversation));
        int conversationTokens = history.Messages.Sum(message => model.CountTokens(message.Text)) + model.CountTokens(text);
        int available = ContextBudget.Available(model.ContextLength, ContextBudget.AnswerReserve(model.ContextLength, (int)MaxTokens), conversationTokens);
        var fitted = new Dictionary<string, AttachmentRecord>();
        for (int i = 0; i < documents.Count; i++)
        {
            int share = available / (documents.Count - i);
            var (fit, tokens, truncated) = ContextBudget.Fit(documents[i].Text ?? "", share, model.CountTokens);
            fitted[documents[i].Id] = documents[i] with { Text = fit, TokenCount = tokens, Truncated = truncated };
            available -= tokens;
        }

        if (fitted.Values.Any(record => record.Truncated))
        {
            Notice = "A document was shortened to fit the context window. Load the model with a longer context to send all of it.";
        }

        return [.. records.Select(record => fitted.GetValueOrDefault(record.Id, record))];
    }

    /// <summary>Synchronizes <see cref="Messages"/> with the active branch, keeping the views of unchanged messages.</summary>
    private void RebuildMessages()
    {
        var path = ConversationTree.ActivePath(_conversation).Where(message => message.Role != ChatRole.System).ToList();
        int common = 0;
        while (common < path.Count && common < Messages.Count && Messages[common].Id == path[common].Id)
        {
            common++;
        }

        while (Messages.Count > common)
        {
            Messages.RemoveAt(Messages.Count - 1);
        }

        foreach (var message in path.Skip(common))
        {
            Messages.Add(new MessageViewModel(message, 1, 1));
        }

        foreach (var view in Messages)
        {
            var siblings = ConversationTree.Siblings(_conversation, view.Record);
            view.VersionCount = siblings.Count;
            view.VersionIndex = siblings.ToList().FindIndex(sibling => sibling.Id == view.Id) + 1;
        }

        OnPropertyChanged(nameof(IsEmpty));
    }

    private void UpdateContextInfo()
    {
        var model = _currentModel();
        var last = ConversationTree.ActivePath(_conversation).LastOrDefault(message => message.Stats is not null)?.Stats;
        int used = last is null ? 0 : last.PromptTokens + last.CompletionTokens;
        if (model is null)
        {
            ContextText = used > 0 ? $"{ByteFormat.Count(used)} tokens" : "";
            ContextFraction = 0;
            return;
        }

        ContextText = $"{ByteFormat.Count(used)} / {ByteFormat.Count(model.ContextLength)} tokens";
        ContextFraction = Math.Clamp((double)used / Math.Max(1, model.ContextLength), 0, 1);
    }

    private void OnComposerChanged()
    {
        OnPropertyChanged(nameof(HasAttachments));
        OnPropertyChanged(nameof(ImageNotSupported));
        SendCommand.NotifyCanExecuteChanged();
    }

    private bool SelectedModelHas(ModelCapabilities capability) => _currentModel() is { } model && model.Capabilities.HasFlag(capability);

    private int EstimateTokens(string text) => _currentModel() is { } model ? model.CountTokens(text) : text.Length / 4;

    private void ApplySampling(SamplingSettings sampling)
    {
        Temperature = sampling.Temperature;
        TopP = sampling.TopP;
        TopK = sampling.TopK;
        MinP = sampling.MinP;
        RepetitionPenalty = sampling.RepetitionPenalty;
        MaxTokens = sampling.MaxTokens;
        SeedText = sampling.Seed?.ToString(CultureInfo.InvariantCulture) ?? "";
    }

    private SamplingSettings CurrentSampling()
    {
        return new SamplingSettings
        {
            Temperature = Math.Round(Temperature, 2),
            TopP = Math.Round(TopP, 2),
            TopK = (int)TopK,
            MinP = Math.Round(MinP, 2),
            RepetitionPenalty = Math.Round(RepetitionPenalty, 2),
            MaxTokens = (int)MaxTokens,
            Seed = int.TryParse(SeedText, NumberStyles.Integer, CultureInfo.InvariantCulture, out var seed) ? seed : null,
        };
    }

    private void SaveSettings()
    {
        if (_loading)
        {
            return;
        }

        _conversation = _conversation with { SystemPrompt = SystemPrompt, EnableThinking = EnableThinking, Sampling = CurrentSampling() };
        _ = SaveAsync();
    }

    private async Task SaveAsync()
    {
        _changed(_conversation);
        await _services.Conversations.SaveAsync(_conversation);
    }

    private static string MediaTypeOf(string path)
    {
        return Path.GetExtension(path).ToLowerInvariant() switch
        {
            ".jpg" or ".jpeg" => "image/jpeg",
            ".webp" => "image/webp",
            ".bmp" => "image/bmp",
            ".gif" => "image/gif",
            _ => "image/png",
        };
    }
}
