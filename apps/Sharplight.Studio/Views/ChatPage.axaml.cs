using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Input.Platform;
using Avalonia.Interactivity;
using Avalonia.Media.Imaging;
using Avalonia.Platform.Storage;
using Sharplight.Studio.ViewModels.Chat;

namespace Sharplight.Studio.Views;

/// <summary>
/// The chat page. Code-behind handles what bindings cannot: Enter to send, pasting images and files, dropping files on the
/// composer, renaming on Enter, and keeping the newest message in view while it streams (unless the user scrolled up).
/// </summary>
internal sealed partial class ChatPage : UserControl
{
    private const double StickDistance = 80;
    private bool _stickToBottom = true;

    public ChatPage()
    {
        InitializeComponent();
        Input.AddHandler(KeyDownEvent, OnInputKeyDown, RoutingStrategies.Tunnel);
        Composer.AddHandler(DragDrop.DragOverEvent, OnDragOver);
        Composer.AddHandler(DragDrop.DragLeaveEvent, (_, _) => Composer.Classes.Remove("drop"));
        Composer.AddHandler(DragDrop.DropEvent, OnDrop);
        MessagesScroll.ScrollChanged += OnScrollChanged;
    }

    private ChatPageViewModel? ViewModel => DataContext as ChatPageViewModel;

    private void OnScrollChanged(object? sender, ScrollChangedEventArgs e)
    {
        var scroll = MessagesScroll;
        if (e.ExtentDelta.Y != 0 || e.ViewportDelta.Y != 0)
        {
            // Content grew (streaming) or the window resized: follow when the user was at the bottom.
            if (_stickToBottom)
            {
                scroll.ScrollToEnd();
            }

            return;
        }

        _stickToBottom = scroll.Offset.Y + scroll.Viewport.Height >= scroll.Extent.Height - StickDistance;
    }

    private void OnInputKeyDown(object? sender, KeyEventArgs e)
    {
        if (ViewModel?.Current is not { } conversation)
        {
            return;
        }

        if (e.Key == Key.Enter && !e.KeyModifiers.HasFlag(KeyModifiers.Shift))
        {
            e.Handled = true;
            if (conversation.SendCommand.CanExecute(null))
            {
                _stickToBottom = true;
                conversation.SendCommand.Execute(null);
            }
        }
        else if (e.Key == Key.V && (e.KeyModifiers.HasFlag(KeyModifiers.Control) || e.KeyModifiers.HasFlag(KeyModifiers.Meta)))
        {
            e.Handled = true;
            _ = PasteAsync(conversation);
        }
    }

    private async Task PasteAsync(ConversationViewModel conversation)
    {
        if (TopLevel.GetTopLevel(this)?.Clipboard is not { } clipboard)
        {
            return;
        }

        var files = await clipboard.TryGetFilesAsync();
        var paths = files?.Select(file => file.TryGetLocalPath()).OfType<string>().ToList() ?? [];
        if (paths.Count > 0)
        {
            await conversation.AddFilesAsync(paths);
            return;
        }

        if (await clipboard.TryGetBitmapAsync() is { } bitmap)
        {
            using var stream = new MemoryStream();
            bitmap.Save(stream, new PngBitmapEncoderOptions());
            conversation.AddImage(stream.ToArray(), "image/png", $"pasted-{DateTime.Now:HHmmss}.png");
            return;
        }

        if (await clipboard.TryGetTextAsync() is { Length: > 0 } text)
        {
            int start = Math.Min(Input.SelectionStart, Input.SelectionEnd);
            int end = Math.Max(Input.SelectionStart, Input.SelectionEnd);
            var current = Input.Text ?? "";
            start = Math.Clamp(start, 0, current.Length);
            end = Math.Clamp(end, start, current.Length);
            Input.Text = current[..start] + text + current[end..];
            Input.CaretIndex = start + text.Length;
        }
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        e.DragEffects = e.DataTransfer.Contains(DataFormat.File) ? DragDropEffects.Copy : DragDropEffects.None;
        Composer.Classes.Add("drop");
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        Composer.Classes.Remove("drop");
        if (ViewModel?.Current is not { } conversation || e.DataTransfer.TryGetFiles() is not { } files)
        {
            return;
        }

        var paths = files.Select(file => file.TryGetLocalPath()).OfType<string>().ToList();
        if (paths.Count > 0)
        {
            await conversation.AddFilesAsync(paths);
        }
    }

    private void OnRenameKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && sender is TextBox { DataContext: ConversationItemViewModel item })
        {
            e.Handled = true;
            ViewModel?.CommitRenameCommand.Execute(item);
        }
    }

    private void OnRenameLostFocus(object? sender, RoutedEventArgs e)
    {
        if (sender is TextBox { DataContext: ConversationItemViewModel { IsRenaming: true } item })
        {
            ViewModel?.CommitRenameCommand.Execute(item);
        }
    }

    /// <inheritdoc/>
    protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
    {
        base.OnAttachedToVisualTree(e);
        _stickToBottom = true;
    }
}
