using Sharplight.Gguf;

namespace Sharplight.GgufDump;

/// <summary>Writes a human-readable description of a GGUF model: header, metadata, optional tensor list and a size histogram per tensor type.</summary>
internal sealed class GgufReport
{
    private readonly GgufModelFiles _model;
    private readonly TextWriter _output;

    public GgufReport(GgufModelFiles model, TextWriter output)
    {
        _model = model;
        _output = output;
    }

    public void Write(bool includeTensors)
    {
        WriteHeader();
        WriteMetadata();
        if (includeTensors)
        {
            WriteTensors();
        }

        WriteTypeHistogram();
    }

    private void WriteHeader()
    {
        var first = _model.First;
        _output.WriteLine($"File: {first.Path}");
        _output.WriteLine($"Version: {first.Version}  shards: {_model.Files.Count}  alignment: {first.Alignment}  data offset: {first.DataOffset}");
        _output.WriteLine($"Tensors: {_model.Tensors.Count}  KVs: {_model.Metadata.Count}");
        _output.WriteLine();
    }

    private void WriteMetadata()
    {
        _output.WriteLine("Metadata:");
        foreach (var (key, value) in _model.Metadata)
        {
            _output.WriteLine($"  {key} = {GgufValueFormatter.Format(value)}");
        }
    }

    private void WriteTensors()
    {
        _output.WriteLine();
        _output.WriteLine("Tensors:");
        foreach (var tensor in _model.Tensors)
        {
            _output.WriteLine($"  {tensor.Name,-48} {tensor.DataType,-8} {tensor.Shape,-22} {tensor.ByteSize,14:N0} B  @{tensor.Offset}");
        }
    }

    private void WriteTypeHistogram()
    {
        _output.WriteLine();
        _output.WriteLine("Tensor type histogram:");
        var byType = _model.Tensors
            .GroupBy(tensor => tensor.DataType)
            .Select(group => (Type: group.Key, Count: group.Count(), Bytes: group.Sum(tensor => tensor.ByteSize)))
            .OrderByDescending(entry => entry.Bytes);
        foreach (var (type, count, bytes) in byType)
        {
            _output.WriteLine($"  {type,-8} count={count,5}  bytes={bytes,16:N0}");
        }

        _output.WriteLine($"Total tensor bytes: {_model.Tensors.Sum(tensor => tensor.ByteSize):N0}");
    }
}
