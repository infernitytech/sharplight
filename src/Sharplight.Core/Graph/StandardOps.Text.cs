using Sharplight.Tensors;

namespace Sharplight.Graph;

// Text: features over token ids and string tensors (DataType.Str, handles into RunParameters.Strings; host-only).
public static partial class StandardOps
{
    /// <summary>
    /// ONNX TfIdfVectorizer over integer token ids: x i32 <c>[C]</c> or <c>[N, C]</c> → f32 <c>[F]</c> or <c>[N, F]</c> n-gram features.
    /// Attributes: <c>min_gram</c>, <c>max_gram</c>, <c>max_skip</c> (long), <c>pool</c> (long[]: the n-grams, grouped by length),
    /// <c>ngram_counts</c> (long[]: start of each length's group in <c>pool</c>, lengths 1, 2, ...), <c>ngram_indexes</c> (long[]: output
    /// column of each pooled n-gram, in pool order), <c>weights</c> (float[], optional, per column), <c>mode</c> ("TF", "IDF", "TFIDF"),
    /// <c>out_features</c> (long: F). N-grams of length ≥ 2 are counted for every skip distance 1..max_skip+1, unigrams once.
    /// </summary>
    public static readonly OpSchema TfIdf = new("TfIdf", OpCategory.Data, 1, 1, (i, a) =>
    {
        var x = i[0].Shape;
        long features = a.Get<long>("out_features");
        return x.Rank switch
        {
            1 => [new(DataType.F32, new Shape(features))],
            2 => [new(DataType.F32, new Shape(x[0], features))],
            _ => throw new ArgumentException($"TfIdf expects x [C] or [N, C], got {x}."),
        };
    });

    /// <summary>A constant string tensor: no inputs → str <c>shape</c> (long[]) holding <c>strings</c> (string[], row-major).</summary>
    public static readonly OpSchema StringConst = new("StringConst", OpCategory.Data, 0, 0, (i, a) =>
        [new(DataType.Str, new Shape(a.Get<long[]>("shape").Select(d => (Dim)d).ToArray()))]);

    /// <summary>
    /// Broadcasting binary op on two str tensors: <c>mode</c> "concat" → str (a + b), "equal" → f32 1/0 (ordinal equality).
    /// </summary>
    public static readonly OpSchema StringBinary = new("StringBinary", OpCategory.Data, 2, 2, (i, a) =>
    {
        var shape = BroadcastShapes(i[0].Shape, i[1].Shape);
        return [new(a.Get<string>("mode") == "equal" ? DataType.F32 : DataType.Str, shape)];
    });

    /// <summary>
    /// ONNX StringNormalizer on str <c>[N]</c>: drops <c>stopwords</c> (string[]; compared ordinally, or case-insensitively when
    /// <c>case_sensitive</c> is false), then changes case (<c>case</c>: "NONE", "LOWER", "UPPER"; invariant culture) → str <c>[count]</c>, where
    /// <c>count_symbol</c> is the dynamic symbol of the number of kept strings, or 1 (one empty string) when none is kept.
    /// </summary>
    public static readonly OpSchema StringNormalize = new("StringNormalize", OpCategory.Data, 1, 1, (i, a) =>
        i[0].Rank == 1 ? [new(DataType.Str, new Shape(Dim.Sym(a.Get<string>("count_symbol"))))]
            : throw new ArgumentException($"StringNormalize expects str [N], got {i[0].Shape}."));

    /// <summary>Whether each string of x (str, any shape) fully matches the regular expression <c>pattern</c> → f32 1/0, same shape.</summary>
    public static readonly OpSchema RegexMatch = new("RegexMatch", OpCategory.Data, 1, 1, (i, a) => [new(DataType.F32, i[0].Shape)]);

    /// <summary>
    /// Converts between strings and numbers, same shape: str → <c>type</c> ("f32" or "i32"; invariant-culture parse, "INF", "-INF" and "NaN"
    /// accepted, unparsable text gives NaN or 0), or f32 / i32 → str (shortest round-trip text; <c>type</c> "str").
    /// </summary>
    public static readonly OpSchema StringCast = new("StringCast", OpCategory.Data, 1, 1, (i, a) => [new(DataType.Get(a.Get<string>("type")), i[0].Shape)]);
}
