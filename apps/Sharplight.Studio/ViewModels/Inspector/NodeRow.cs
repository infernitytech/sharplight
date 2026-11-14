using Sharplight.Package;
using Sharplight.Studio.Services.Inspection;

namespace Sharplight.Studio.ViewModels.Inspector;

/// <summary>One row of the node list; the fields are formatted once when the list is built.</summary>
/// <param name="Info">The node.</param>
/// <param name="Name">Names of the node's outputs.</param>
/// <param name="Inputs">Input values as <c>%id:name type[shape]</c>.</param>
/// <param name="Outputs">Output types and shapes.</param>
internal sealed record NodeRow(NodeInfo Info, string Name, string Inputs, string Outputs) : IFilterableRow
{
    /// <summary>Prefix of a filter term that matches the op name exactly (<c>op:MatMul</c>).</summary>
    public const string OpTermPrefix = "op:";

    /// <summary>Node id.</summary>
    public int Id => Info.Id;

    /// <summary>Op name.</summary>
    public string Op => Info.Op;

    /// <summary>Placement group (layer).</summary>
    public string Group => Info.Group;

    /// <summary>Attributes as <c>{key=value, ...}</c>.</summary>
    public string Attributes => Info.Attributes;

    /// <summary>The row of a node.</summary>
    public static NodeRow From(NodeInfo node) => new(
        node,
        string.Join(", ", node.Outputs.Select(o => o.Name)),
        string.Join(", ", node.Inputs),
        string.Join(", ", node.Outputs.Select(o => $"{o.Type}{o.Shape}")));

    /// <inheritdoc/>
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
