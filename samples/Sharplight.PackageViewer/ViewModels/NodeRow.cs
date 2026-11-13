using Sharplight.Package;

namespace Sharplight.PackageViewer.ViewModels;

/// <summary>One row of the node table; the fields are formatted once when the table is built.</summary>
/// <param name="Id">Node id.</param>
/// <param name="Op">Op name.</param>
/// <param name="Name">Names of the node's outputs.</param>
/// <param name="Group">Placement group (layer).</param>
/// <param name="Inputs">Input values as <c>%id:name type[shape]</c>.</param>
/// <param name="Outputs">Output types and shapes.</param>
/// <param name="Attributes">Attributes as <c>{key=value, ...}</c>.</param>
public sealed record NodeRow(int Id, string Op, string Name, string Group, string Inputs, string Outputs, string Attributes) : IFilterableRow
{
    public static NodeRow From(NodeInfo node) => new(
        node.Id,
        node.Op,
        string.Join(", ", node.Outputs.Select(o => o.Name)),
        node.Group,
        string.Join(", ", node.Inputs),
        string.Join(", ", node.Outputs.Select(o => $"{o.Type}{o.Shape}")),
        node.Attributes);

    /// <summary>Prefix of a filter term that matches the op name exactly (<c>op:MatMul</c>).</summary>
    public const string OpTermPrefix = "op:";

    public bool Matches(string[] terms)
    {
        foreach (var term in terms)
        {
            bool match = term.StartsWith(OpTermPrefix, StringComparison.OrdinalIgnoreCase)
                ? Op.Equals(term[OpTermPrefix.Length..], StringComparison.OrdinalIgnoreCase)
                : TextFilter.MatchesTerm(term, Op, Name, Group, Inputs, Outputs, Attributes);
            if (!match)
            {
                return false;
            }
        }

        return true;
    }
}
