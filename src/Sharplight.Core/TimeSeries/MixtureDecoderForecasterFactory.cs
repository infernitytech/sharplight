using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>Creates <see cref="ITimeSeriesForecaster"/>s for <see cref="ModelKinds.TimeSeriesForecasting"/> models of the mixture-decoder form (Moirai-MoE).</summary>
internal sealed class MixtureDecoderForecasterFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.TimeSeriesForecasting;

    public Type ClientType => typeof(ITimeSeriesForecaster);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Decoder
            && c.ConfigValue(TimeSeriesGraphs.FormKey, "") == TimeSeriesGraphs.MixtureDecoderForm);

    public ITaskClient Create(TaskModel model) => new MixtureDecoderForecasterClient(model);
}
