namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>A node that reads an output of the selected node.</summary>
/// <param name="Id">Node id.</param>
/// <param name="Op">Op name.</param>
/// <param name="Group">Placement group (layer).</param>
internal sealed record NodeLinkRow(int Id, string Op, string Group)
{
    /// <summary><c>#12 MatMul</c>.</summary>
    public string Label => $"#{Id} {Op}";
}
