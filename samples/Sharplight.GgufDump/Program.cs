using Sharplight.Gguf;

if (args.Length == 0 || args[0] is "-h" or "--help")
{
    Console.WriteLine("Usage: GgufDump <file.gguf> [--tensors]");
    return 1;
}

bool showTensors = args.Contains("--tensors");
using var model = GgufModelFiles.Open(args[0]);
var f = model.First;
Console.WriteLine($"File: {f.Path}");
Console.WriteLine($"Version: {f.Version}  shards: {model.Files.Count}  alignment: {f.Alignment}  data offset: {f.DataOffset}");
Console.WriteLine($"Tensors: {model.Tensors.Count}  KVs: {model.Metadata.Count}");
Console.WriteLine();
Console.WriteLine("Metadata:");
foreach (var (k, v) in model.Metadata)
{
    Console.WriteLine($"  {k} = {Format(v)}");
}

if (showTensors)
{
    Console.WriteLine();
    Console.WriteLine("Tensors:");
    foreach (var t in model.Tensors)
    {
        Console.WriteLine($"  {t.Name,-48} {t.DataType,-8} {t.Shape,-22} {t.ByteSize,14:N0} B  @{t.Offset}");
    }
}

Console.WriteLine();
Console.WriteLine("Tensor type histogram:");
foreach (var g in model.Tensors.GroupBy(t => t.DataType).OrderByDescending(g => g.Sum(t => t.ByteSize)))
{
    Console.WriteLine($"  {g.Key,-8} count={g.Count(),5}  bytes={g.Sum(t => t.ByteSize),16:N0}");
}

Console.WriteLine($"Total tensor bytes: {model.Tensors.Sum(t => t.ByteSize):N0}");
return 0;

static string Format(object v, int depth = 0)
{
    switch (v)
    {
        case string s: return s.Length > 100 ? $"\"{Escape(s[..100])}\"... ({s.Length} chars)" : $"\"{Escape(s)}\"";
        case Array a:
            int n = Math.Min(a.Length, 8);
            var items = new List<string>();
            for (int i = 0; i < n; i++)
            {
                items.Add(Format(a.GetValue(i)!, depth + 1));
            }

            return $"{a.GetType().GetElementType()!.Name}[{a.Length}] [{string.Join(", ", items)}{(a.Length > n ? ", ..." : "")}]";
        default: return Convert.ToString(v, System.Globalization.CultureInfo.InvariantCulture) ?? "";
    }
}

static string Escape(string s) => s.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
