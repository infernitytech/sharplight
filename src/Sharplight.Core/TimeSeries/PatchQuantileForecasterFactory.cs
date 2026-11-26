using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>Creates <see cref="ITimeSeriesForecaster"/>s for <see cref="ModelKinds.TimeSeriesForecasting"/> models of the patch-quantile form (<see cref="TimeSeriesGraphs.ForecastGraph"/>).</summary>
internal sealed class PatchQuantileForecasterFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.TimeSeriesForecasting;

    public Type ClientType => typeof(ITimeSeriesForecaster);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Encoder && c.Graphs.Contains(TimeSeriesGraphs.ForecastGraph));

    public ITaskClient Create(TaskModel model) => new PatchQuantileForecasterClient(model);
}
