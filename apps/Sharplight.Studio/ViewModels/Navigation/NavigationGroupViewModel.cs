using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.Navigation;

/// <summary>A root of the navigation (a modality such as Text or Images, or the models): a collapsible header over its entries.</summary>
internal sealed partial class NavigationGroupViewModel : ObservableObject
{
    /// <param name="id">The stable id, under which the collapsed state is remembered in the settings.</param>
    /// <param name="title">The header's label.</param>
    /// <param name="icon">Icon resource key (a geometry in <c>Icons.axaml</c>).</param>
    /// <param name="items">The entries, in the order shown.</param>
    public NavigationGroupViewModel(string id, string title, string icon, IReadOnlyList<NavigationItemViewModel> items)
    {
        Id = id;
        Title = title;
        Icon = icon;
        Items = items;
        foreach (var item in items)
        {
            item.Group = this;
            item.PropertyChanged += (_, args) =>
            {
                if (args.PropertyName == nameof(NavigationItemViewModel.Badge))
                {
                    Badge = Items.Sum(entry => entry.Badge);
                }
            };
        }

        Badge = items.Sum(entry => entry.Badge);
    }

    /// <summary>The stable id (settings key).</summary>
    public string Id { get; }

    /// <summary>The header's label.</summary>
    public string Title { get; }

    /// <summary>Icon resource key.</summary>
    public string Icon { get; }

    /// <summary>The entries.</summary>
    public IReadOnlyList<NavigationItemViewModel> Items { get; }

    /// <summary>The entries are shown below the header.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowsCollapsedBadge))]
    public partial bool IsExpanded { get; set; } = true;

    /// <summary>The sum of the entries' badges, shown on the header while collapsed and on the rail icon; 0 hides it.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ShowsCollapsedBadge))]
    public partial int Badge { get; private set; }

    /// <summary>The header shows <see cref="Badge"/>: the group is collapsed, so its entries' badges are hidden.</summary>
    public bool ShowsCollapsedBadge => !IsExpanded && Badge > 0;

    /// <summary>One of the entries shows the visible page (the header is highlighted, which matters while collapsed and in the rail).</summary>
    [ObservableProperty]
    public partial bool ContainsCurrent { get; set; }

    /// <summary>Flips <see cref="IsExpanded"/>.</summary>
    public void Toggle() => IsExpanded = !IsExpanded;
}
