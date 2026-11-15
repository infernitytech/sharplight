using System.Globalization;

namespace Sharplight.Tasks;

/// <summary>The service key of the <see cref="TaskModel"/> behind a client registered with <c>services.AddTaskClient&lt;TClient&gt;</c>.</summary>
/// <param name="ClientType">The client interface.</param>
/// <param name="ServiceKey">The client's service key, or <see langword="null"/>.</param>
internal sealed record TaskModelKey(Type ClientType, object? ServiceKey)
{
    /// <summary>The name of the <see cref="TaskLoadOptions"/> instance configured for this model.</summary>
    public string OptionsName => ServiceKey is null
        ? ClientType.FullName!
        : string.Create(CultureInfo.InvariantCulture, $"{ClientType.FullName}:{ServiceKey}");
}
