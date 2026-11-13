using Sharplight.Tensors;

namespace Sharplight.Graph;

// Text features over token ids.
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
}
