using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels;

/// <summary>A page of the main window.</summary>
internal abstract partial class PageViewModel : ObservableObject
{
    /// <summary>Title shown in the navigation and page header.</summary>
    public abstract string Title { get; }

    /// <summary>Icon resource key (a geometry in <c>Icons.axaml</c>).</summary>
    public abstract string Icon { get; }

    /// <summary>Small count shown next to the navigation item; 0 hides it.</summary>
    [ObservableProperty]
    public partial int Badge { get; set; }

    /// <summary>Called when the page becomes visible.</summary>
    public virtual Task ActivateAsync() => Task.CompletedTask;

    /// <summary>Called when another page becomes visible.</summary>
    public virtual void Deactivate()
    {
    }
}
