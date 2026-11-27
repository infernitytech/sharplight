using Sharplight.Models;
using Sharplight.Tasks;

namespace Sharplight.TimeSeries;

/// <summary>Creates <see cref="ITimeSeriesForecaster"/>s for <see cref="ModelKinds.TimeSeriesForecasting"/> models of the group-encoder form (Chronos-2).</summary>
internal sealed class GroupEncoderForecasterFactory : ITaskClientFactory
{
    public string Kind => ModelKinds.TimeSeriesForecasting;

    public Type ClientType => typeof(ITimeSeriesForecaster);

    public bool CanCreate(TaskModel model)
        => model.Description.Components.Any(c => c.Role == ModelComponentRoles.Encoder
            && c.ConfigValue(TimeSeriesGraphs.FormKey, "") == TimeSeriesGraphs.GroupEncoderForm);

    public ITaskClient Create(TaskModel model) => new GroupEncoderForecasterClient(model);
}
