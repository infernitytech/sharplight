using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using Sharplight.Package;
using Sharplight.Studio.Models;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>The weights tab: the weight index (filterable and sortable) and the totals per stored type.</summary>
internal sealed partial class WeightsViewModel : ObservableObject
{
    private readonly IReadOnlyList<WeightRow> _all;

    /// <param name="weights">The weight index.</param>
    /// <param name="dataBytes">Length of the weight data section.</param>
    public WeightsViewModel(IReadOnlyList<PackageWeightInfo> weights, long dataBytes)
    {
        _all = [.. weights.Select(WeightRow.From).OrderBy(w => w.Offset)];
        long total = _all.Sum(w => w.Bytes);
        double denominator = Math.Max(1, total);
        TypeTotals = [.. _all.GroupBy(w => w.Type)
            .Select(g => new ShareRow(g.Key, $"{g.Count():N0} weights", g.Sum(w => w.Bytes), 100 * g.Sum(w => w.Bytes) / denominator))
            .OrderByDescending(t => t.Bytes)];
        long elements = _all.Sum(w => w.Elements);
        Totals = $"{_all.Count:N0} weights · {ByteFormat.Format(total)} · {ByteFormat.Count(elements)} parameters · weight data section {ByteFormat.Format(dataBytes)}";
        Weights = _all;
        Summary = Describe(_all);
    }

    /// <summary>Bytes per stored type, largest first.</summary>
    public IReadOnlyList<ShareRow> TypeTotals { get; }

    /// <summary>Count, size and parameters of all weights.</summary>
    public string Totals { get; }

    /// <summary>Filter of the weight list (whitespace-separated terms, all must match name, type or shape).</summary>
    [ObservableProperty]
    public partial string? Filter { get; set; }

    /// <summary>The weights matching <see cref="Filter"/>, in <see cref="SortColumn"/> order.</summary>
    [ObservableProperty]
    public partial IReadOnlyList<WeightRow> Weights { get; private set; }

    /// <summary>Count and size of the shown weights.</summary>
    [ObservableProperty]
    public partial string Summary { get; private set; }

    /// <summary>The selected weight.</summary>
    [ObservableProperty]
    public partial WeightRow? SelectedWeight { get; set; }

    /// <summary>The sort column.</summary>
    [ObservableProperty]
    public partial WeightColumn SortColumn { get; private set; } = WeightColumn.Offset;

    /// <summary>Largest (or last) first.</summary>
    [ObservableProperty]
    public partial bool SortDescending { get; private set; }

    /// <summary>Header of the name column, with the sort direction when it is the sort column.</summary>
    public string NameHeader => Header(WeightColumn.Name, "Name");

    /// <summary>Header of the type column.</summary>
    public string TypeHeader => Header(WeightColumn.Type, "Type");

    /// <summary>Header of the shape column.</summary>
    public string ShapeHeader => Header(WeightColumn.Shape, "Shape");

    /// <summary>Header of the bits-per-element column.</summary>
    public string BitsHeader => Header(WeightColumn.Bits, "Bits");

    /// <summary>Header of the size column.</summary>
    public string SizeHeader => Header(WeightColumn.Size, "Size");

    /// <summary>Header of the offset column.</summary>
    public string OffsetHeader => Header(WeightColumn.Offset, "Offset");

    /// <summary>Sorts by a column; sorting by the same column again reverses the order.</summary>
    [RelayCommand]
    public void SortBy(string column)
    {
        if (!Enum.TryParse<WeightColumn>(column, out var value))
        {
            return;
        }

        SortDescending = value == SortColumn ? !SortDescending : value is WeightColumn.Size or WeightColumn.Bits or WeightColumn.Shape;
        SortColumn = value;
        Refresh();
        OnPropertyChanged(nameof(NameHeader));
        OnPropertyChanged(nameof(TypeHeader));
        OnPropertyChanged(nameof(ShapeHeader));
        OnPropertyChanged(nameof(BitsHeader));
        OnPropertyChanged(nameof(SizeHeader));
        OnPropertyChanged(nameof(OffsetHeader));
    }

    partial void OnFilterChanged(string? value) => Refresh();

    private void Refresh()
    {
        var shown = TextFilter.Apply(_all, Filter);
        IEnumerable<WeightRow> ordered = SortColumn switch
        {
            WeightColumn.Name => shown.OrderBy(w => w.Name, StringComparer.Ordinal),
            WeightColumn.Type => shown.OrderBy(w => w.Type, StringComparer.Ordinal).ThenBy(w => w.Offset),
            WeightColumn.Shape => shown.OrderBy(w => w.Elements).ThenBy(w => w.Offset),
            WeightColumn.Bits => shown.OrderBy(w => w.BitsPerElement).ThenBy(w => w.Offset),
            WeightColumn.Size => shown.OrderBy(w => w.Bytes).ThenBy(w => w.Offset),
            _ => shown.OrderBy(w => w.Offset),
        };
        if (SortDescending)
        {
            ordered = ordered.Reverse();
        }

        Weights = [.. ordered];
        Summary = Describe(Weights);
    }

    private string Header(WeightColumn column, string text) => column != SortColumn ? text : SortDescending ? text + " ↓" : text + " ↑";

    private string Describe(IReadOnlyList<WeightRow> shown)
    {
        var size = ByteFormat.Format(shown.Sum(w => w.Bytes));
        return shown.Count == _all.Count ? $"{_all.Count:N0} weights, {size}" : $"Showing {shown.Count:N0} of {_all.Count:N0} weights, {size}";
    }
}
