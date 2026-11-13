using CommunityToolkit.Mvvm.ComponentModel;
using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>The weights tab: the weight index (filterable, sortable in the view) and totals per element type.</summary>
public sealed partial class WeightsViewModel : ObservableObject
{
    private readonly IReadOnlyList<WeightRow> _all;

    public WeightsViewModel(IReadOnlyList<PackageWeightInfo> weights)
    {
        _all = [.. weights.Select(WeightRow.From)];
        double total = Math.Max(1, _all.Sum(w => w.Bytes));
        TypeTotals = [.. _all.GroupBy(w => w.Type)
            .Select(g => new TypeTotalRow(g.Key, g.Count(), g.Sum(w => w.Bytes), 100 * g.Sum(w => w.Bytes) / total))
            .OrderByDescending(t => t.Bytes)];
        Weights = _all;
        Summary = Describe(_all);
    }

    public IReadOnlyList<TypeTotalRow> TypeTotals { get; }

    /// <summary>Filter of the weight table (whitespace-separated terms, all must match name, type or shape).</summary>
    [ObservableProperty]
    public partial string? Filter { get; set; }

    [ObservableProperty]
    public partial IReadOnlyList<WeightRow> Weights { get; private set; }

    /// <summary>Count and size of the shown weights.</summary>
    [ObservableProperty]
    public partial string Summary { get; private set; }

    partial void OnFilterChanged(string? value)
    {
        Weights = TextFilter.Apply(_all, value);
        Summary = Describe(Weights);
    }

    private string Describe(IReadOnlyList<WeightRow> shown)
    {
        var size = ByteSizeFormat.Format(shown.Sum(w => w.Bytes));
        return shown.Count == _all.Count ? $"{_all.Count:N0} weights, {size}" : $"Showing {shown.Count:N0} of {_all.Count:N0} weights, {size}";
    }
}
