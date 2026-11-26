using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>Creates <see cref="ITimeSeriesForecaster"/>s for <see cref="ModelKinds.TimeSeriesForecasting"/> models of the token form (an encoder and a decoder component).</summary>
internal sealed class TokenForecasterFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.TimeSeriesForecasting;

    public Type ClientType => typeof(ITimeSeriesForecaster);

    public bool CanCreate(TaskModel model)
        => model.Description.TryGetComponent(ModelComponentRoles.Encoder, out var encoder)
           && encoder.ConfigValue(TimeSeriesGraphs.FormKey, "") == TimeSeriesGraphs.TokenForm
           && model.Description.TryGetComponent(ModelComponentRoles.Decoder, out _);

    public ITaskClient Create(TaskModel model) => new TokenForecasterClient(model);
}
