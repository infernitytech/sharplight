using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>Creates <see cref="ITimeSeriesForecaster"/>s for <see cref="ModelKinds.TimeSeriesForecasting"/> models of the patch-decoder form (<see cref="TimeSeriesGraphs.PatchDecoderGraph"/>).</summary>
internal sealed class PatchDecoderForecasterFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.TimeSeriesForecasting;

    public Type ClientType => typeof(ITimeSeriesForecaster);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Decoder && c.Graphs.Contains(TimeSeriesGraphs.PatchDecoderGraph));

    public ITaskClient Create(TaskModel model) => new PatchDecoderForecasterClient(model);
}
