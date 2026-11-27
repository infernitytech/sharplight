using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>Creates <see cref="ITimeSeriesForecaster"/>s for <see cref="ModelKinds.TimeSeriesForecasting"/> models of the masked-encoder form (Moirai 1.x).</summary>
internal sealed class MaskedEncoderForecasterFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.TimeSeriesForecasting;

    public Type ClientType => typeof(ITimeSeriesForecaster);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Encoder
            && c.ConfigValue(TimeSeriesGraphs.FormKey, "") == TimeSeriesGraphs.MaskedEncoderForm);

    public ITaskClient Create(TaskModel model) => new MaskedEncoderForecasterClient(model);
}
