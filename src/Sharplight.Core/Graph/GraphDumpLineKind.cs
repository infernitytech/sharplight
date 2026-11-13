namespace Sharplight.Graph;

/// <summary>What a line of a graph dump (<see cref="ComputeGraph.DumpLines"/>) describes.</summary>
public enum GraphDumpLineKind
{
    /// <summary><c>graph name</c>: the start of a graph, or of a control-flow subgraph (prefixed with its attribute name).</summary>
    Graph,
    /// <summary><c>input %id:name type[shape] name (role)</c>.</summary>
    Input,
    /// <summary><c>weight %id:name type[shape]</c>: a weight the graph reads.</summary>
    Weight,
    /// <summary><c>state %id:name type[shape] (kind)</c>: a persistent state the graph reads or writes.</summary>
    State,
    /// <summary><c>const %id:name type[shape]</c>: a constant embedded in the graph.</summary>
    Constant,
    /// <summary><c>%out:name type[shape], ... = Op(%in, ...) {attributes}  @group</c>.</summary>
    Node,
    /// <summary><c>output name = %id</c>.</summary>
    Output,
    /// <summary><c>dynamic symbol &lt;= bound</c>.</summary>
    DynamicSymbol,
}
