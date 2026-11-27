using CommunityToolkit.Mvvm.ComponentModel;

namespace Sharplight.Studio.ViewModels.TimeSeries;

/// <summary>A numeric column of the input that can serve as a covariate of the forecast (models with covariate support only).</summary>
internal sealed partial class CovariateColumnViewModel : ObservableObject
{
    /// <param name="name">The column header.</param>
    public CovariateColumnViewModel(string name) => Name = name;

    /// <summary>The column header.</summary>
    public string Name { get; }

    /// <summary>The column is used as a covariate.</summary>
    [ObservableProperty]
    public partial bool IsSelected { get; set; }

    /// <summary>
    /// The column's values after the last observed row of the target are known future values (for example a promotion calendar). Without such rows the column
    /// is a past-only covariate whatever this says.
    /// </summary>
    [ObservableProperty]
    public partial bool KnownInFuture { get; set; }
}
