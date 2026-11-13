# Sharplight Package Viewer

A small Avalonia desktop viewer for Sharplight model packages (`.slm`, see [docs/model-format.md](../../docs/model-format.md)).
It uses only the public inspection API of `Sharplight.Package` (`ModelPackageReader.Open`): no weight bytes and no artifact
entry bytes are read, so even multi-GB packages open in a fraction of a second.

```sh
dotnet run -c Release --project samples/Sharplight.PackageViewer -- model.slm
# or, after a build:
dotnet samples/Sharplight.PackageViewer/bin/Release/net11.0/sharplight-viewer.dll model.slm
```

Open a package with **Open...** (Ctrl+O), by dropping the file on the window, or by passing its path on the command line. The
theme (light/dark) follows the operating system.

| Tab | Shows |
| --- | --- |
| Overview | Format version, file size, bytes `sharplight-convert compact` would reclaim, model, graphs, the section table (kind, offset, length, name), and all model properties and source metadata as a filterable key/value table; the selected value is shown in full below. |
| Graphs | The stored graphs with node counts. For the selected graph: inputs/outputs with types and shapes, the op histogram (select an op to filter the nodes to it), the node table (op, output names, group, output shapes, inputs, attributes; filterable, `op:MatMul` matches an op exactly) and the **IR view**. |
| Weights | The weight index (name, type, shape, size, offset), sortable and filterable, with totals per element type. |
| Artifacts | Stored artifact sets grouped by provider and tier, with entry counts and sizes; the selected set's descriptor and its (filterable) entry list. |

The IR view lists the selected graph in the format of `ComputeGraph.Dump` (`sharplight-onnxrun --dump-graph`): inputs, the
weights, states and constants the graph reads, one line per node with output types, shapes, attributes and group, control-flow
subgraphs (`If` branches) indented below their node, outputs and dynamic symbols. Find text with Enter / Shift+Enter. Selecting
a line shows the producers of the values it reads (`%12 from`), the consumers of the values it defines (`%12 used by`) and a
two-level dataflow neighborhood (`<<` producers, `>>` consumers); selecting any of them jumps to that line, **Back** returns.
Selecting a node in the node table selects its IR line and vice versa.

Node table and IR are built on a background thread when a graph is first selected and are virtualized; a 100,000-node graph
loads in about a second and filters in well under that (see `tests/Sharplight.PackageViewer.Tests`).
