using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>Creates <see cref="ITimeSeriesForecaster"/>s for <see cref="ModelKinds.TimeSeriesForecasting"/> models of the quantile-decoder form (Moirai 2.0).</summary>
internal sealed class QuantileDecoderForecasterFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.TimeSeriesForecasting;

    public Type ClientType => typeof(ITimeSeriesForecaster);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Decoder
            && c.ConfigValue(TimeSeriesGraphs.FormKey, "") == TimeSeriesGraphs.QuantileDecoderForm);

    public ITaskClient Create(TaskModel model) => new QuantileDecoderForecasterClient(model);
}
