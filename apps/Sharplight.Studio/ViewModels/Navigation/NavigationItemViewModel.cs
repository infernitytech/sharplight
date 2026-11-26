using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Sharplight.Studio.ViewModels.Navigation;

/// <summary>
/// One entry of the navigation: a page (or one tab of the Models page) with the label the navigation shows for it, and the model kinds
/// whose models it runs (the routing table of <see cref="NavigationViewModel.ForKind"/>).
/// </summary>
internal sealed partial class NavigationItemViewModel : ObservableObject
{
    private readonly Action<NavigationItemViewModel> _open;

    /// <param name="title">The label in the navigation (the page's header can say more: "Generate" opens the Image page).</param>
    /// <param name="description">What the entry is for, shown as its tooltip and its accessible help text.</param>
    /// <param name="icon">Icon resource key (a geometry in <c>Icons.axaml</c>).</param>
    /// <param name="page">The page it shows.</param>
    /// <param name="tab">The tab of <paramref name="page"/> it shows (the Models page's tabs), or null for the page as it is.</param>
    /// <param name="kinds">The model kinds whose models open on this entry.</param>
    /// <param name="showsBadge">The entry carries the page's <see cref="PageViewModel.Badge"/> (one entry per page does).</param>
    /// <param name="open">Shows the entry (the main window's navigation).</param>
    public NavigationItemViewModel(string title, string description, string icon, PageViewModel page, int? tab, IReadOnlyList<string> kinds,
        bool showsBadge, Action<NavigationItemViewModel> open)
    {
        Title = title;
        Description = description;
        Icon = icon;
        Page = page;
        Tab = tab;
        Kinds = kinds;
        _open = open;
        if (showsBadge)
        {
            Badge = page.Badge;
            page.PropertyChanged += (_, args) =>
            {
                if (args.PropertyName == nameof(PageViewModel.Badge))
                {
                    Badge = page.Badge;
                }
            };
        }
    }

    /// <summary>The label in the navigation.</summary>
    public string Title { get; }

    /// <summary>What the entry is for (tooltip and accessible help text).</summary>
    public string Description { get; }

    /// <summary>Icon resource key.</summary>
    public string Icon { get; }

    /// <summary>The page it shows.</summary>
    public PageViewModel Page { get; }

    /// <summary>The tab of <see cref="Page"/> it shows, or null.</summary>
    public int? Tab { get; }

    /// <summary>The model kinds whose models open on this entry (empty for the pages that run no model).</summary>
    public IReadOnlyList<string> Kinds { get; }

    /// <summary>The group the entry is listed under; null for the app-level entries at the bottom.</summary>
    public NavigationGroupViewModel? Group { get; internal set; }

    /// <summary>The small count shown next to the entry (the page's badge); 0 hides it.</summary>
    [ObservableProperty]
    public partial int Badge { get; private set; }

    /// <summary>The entry shows the visible page (highlighted).</summary>
    [ObservableProperty]
    public partial bool IsCurrent { get; set; }

    /// <summary>Shows the entry's page (and tab).</summary>
    [RelayCommand]
    private void Open() => _open(this);
}
