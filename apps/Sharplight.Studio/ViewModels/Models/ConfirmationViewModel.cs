using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Sharplight.Studio.ViewModels.Models;

/// <summary>A question shown above a list before something is deleted or imported: what happens, how much space it frees, and the two answers.</summary>
internal sealed partial class ConfirmationViewModel : ObservableObject
{
    private readonly Func<Task> _action;
    private readonly Action<ConfirmationViewModel> _dismissed;

    /// <param name="message">What will happen, including the space it frees.</param>
    /// <param name="confirmText">The text of the confirming button.</param>
    /// <param name="isDestructive">The action deletes files (the button is drawn as a danger button).</param>
    /// <param name="action">Carried out on confirmation.</param>
    /// <param name="dismissed">Called when the question is answered either way, so its owner can hide it.</param>
    public ConfirmationViewModel(string message, string confirmText, bool isDestructive, Func<Task> action, Action<ConfirmationViewModel> dismissed)
    {
        Message = message;
        ConfirmText = confirmText;
        IsDestructive = isDestructive;
        _action = action;
        _dismissed = dismissed;
    }

    /// <summary>What will happen.</summary>
    public string Message { get; }

    /// <summary>The text of the confirming button.</summary>
    public string ConfirmText { get; }

    /// <summary>The action deletes files.</summary>
    public bool IsDestructive { get; }

    /// <summary>Carries the action out.</summary>
    [RelayCommand]
    private async Task ConfirmAsync()
    {
        _dismissed(this);
        await _action();
    }

    /// <summary>Drops the question.</summary>
    [RelayCommand]
    private void Cancel() => _dismissed(this);
}
