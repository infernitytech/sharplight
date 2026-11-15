namespace Sharplight.Tasks;

/// <summary>
/// The <see cref="Models.ModelDefinition.Properties"/> keys of the task descriptor (<see cref="ModelTaskDescriptor"/>). Plain typed
/// properties, so every container that stores a definition (the <c>.slm</c> package) stores the descriptor without a format change;
/// docs/model-format.md documents the layout.
/// </summary>
public static class TaskPropertyKeys
{
    /// <summary>Descriptor layout version (<c>i32</c>); see <see cref="ModelTaskDescriptor.CurrentVersion"/>.</summary>
    public const string Version = "task.version";

    /// <summary>Task id (<c>string</c>, see <see cref="ModelTasks"/>).</summary>
    public const string Kind = "task.kind";

    /// <summary>Component roles in declaration order (<c>string[]</c>).</summary>
    public const string Components = "task.components";

    /// <summary>Prefix of the per-component keys: <c>task.component.&lt;role&gt;.graph</c> and <c>task.component.&lt;role&gt;.metadata</c>.</summary>
    public const string ComponentPrefix = "task.component.";

    /// <summary>Suffix of the key holding a component's graph name (<c>string</c>).</summary>
    public const string ComponentGraphSuffix = ".graph";

    /// <summary>Suffix of the key holding a component's source-metadata prefix (<c>string</c>).</summary>
    public const string ComponentMetadataSuffix = ".metadata";

    /// <summary>Prefix of the task settings: <c>task.setting.&lt;name&gt;</c> (any property value type).</summary>
    public const string SettingPrefix = "task.setting.";
}
