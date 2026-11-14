using Sharplight.Graph;
using Sharplight.Tensors;

namespace Sharplight.Vulkan.Tests;

/// <summary>
/// NonMaxSuppression on the GPU against the CPU oracle: selected rows and their data-dependent count, in shared-memory and global
/// sorts, with ties, empty selections, center-point boxes, a symbolic batch and box count, and a consumer of the count in a later
/// segment.
/// </summary>
public sealed class NonMaxSuppressionTests
{
    /// <summary>Corner boxes (y1, x1, y2, x2) clustered around a few centers so that many overlap; some have flipped corners.</summary>
    private static float[] Boxes(long count, int seed, bool centerPoint = false)
    {
        var rng = new Random(seed);
        var boxes = new float[count * 4];
        for (long i = 0; i < count; i++)
        {
            float cy = (rng.Next(6) * 0.15f) + (float)rng.NextDouble() * 0.1f;
            float cx = (rng.Next(6) * 0.15f) + (float)rng.NextDouble() * 0.1f;
            float h = 0.05f + (float)rng.NextDouble() * 0.2f;
            float w = 0.05f + (float)rng.NextDouble() * 0.2f;
            if (centerPoint)
            {
                (boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]) = (cx, cy, w, h);
            }
            else if (rng.Next(4) == 0)
            {
                (boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]) = (cy + h / 2, cx + w / 2, cy - h / 2, cx - w / 2);
            }
            else
            {
                (boxes[i * 4], boxes[i * 4 + 1], boxes[i * 4 + 2], boxes[i * 4 + 3]) = (cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2);
            }
        }

        return boxes;
    }

    private static float[] Scores(long count, int seed, int levels = 0)
    {
        var scores = Rnd.Pos(count, seed, 0f, 1f);
        if (levels > 0)
        {
            for (long i = 0; i < count; i++)
            {
                scores[i] = MathF.Round(scores[i] * levels) / levels;
            }
        }

        return scores;
    }

    private static Value Nms(TG tg, Value boxes, Value scores, Dim bound, long maxPerClass, float iou, float score, bool centerPoint)
    {
        string count = tg.B.DynamicSymbol("nms", bound);
        var attributes = Attributes.From(("count_symbol", count), ("max_per_class", maxPerClass), ("iou_threshold", iou), ("center_point_box", centerPoint));
        if (!float.IsNegativeInfinity(score))
        {
            attributes = attributes.With("score_threshold", score);
        }

        return tg.B.Emit1("NonMaxSuppression", [boxes, scores], attributes);
    }

    [Theory]
    [InlineData(2, 3, 500, 50, 0.5f, 0.3f, false, 0)]      // a few classes
    [InlineData(1, 80, 3000, 200, 0.45f, 0.05f, false, 0)]  // many classes, candidates sorted in shared memory
    [InlineData(1, 2, 12000, 100, 0.5f, 0.01f, false, 0)]   // more candidates than the shared-memory sort holds
    [InlineData(2, 4, 700, 1000, 0.3f, float.NegativeInfinity, false, 0)]   // no score threshold, max beyond the box count
    [InlineData(1, 5, 400, 30, 0.5f, 0.2f, false, 4)]       // tied scores: lower box index first
    [InlineData(2, 3, 600, 40, 0.5f, 0.25f, true, 0)]       // center-point boxes
    [InlineData(1, 3, 300, 0, 0.5f, 0.2f, false, 0)]        // max_per_class 0: nothing selected
    [InlineData(1, 3, 300, 20, 0.5f, 2f, false, 0)]         // no box above the score threshold
    [InlineData(1, 2, 2000, 2000, 1f, 0f, false, 0)]        // IoU threshold 1: nothing suppressed
    public void Classes(int batches, int classes, int boxCount, long maxPerClass, float iou, float score, bool centerPoint, int levels)
    {
        var tg = new TG();
        var boxes = tg.InF("boxes", batches, boxCount, 4);
        var scores = tg.InF("scores", batches, classes, boxCount);
        var selected = Nms(tg, boxes, scores, batches * classes * Math.Min(Math.Max(maxPerClass, 1), boxCount), maxPerClass, iou, score, centerPoint);
        tg.B.Output("selected", selected);
        var step = new Step(1, 0, (boxes, Boxes((long)batches * boxCount, 3, centerPoint)), (scores, Scores((long)batches * classes * boxCount, 5, levels)));
        Diff.Check("NonMaxSuppression", tg.B.Build(), step, atol: 0, rtol: 0,
            what: $"b={batches} c={classes} s={boxCount} max={maxPerClass} iou={iou} score={score} center={centerPoint} levels={levels}");
    }

    /// <summary>Identical boxes and identical scores: every duplicate is suppressed, ties keep the lowest index.</summary>
    [Fact]
    public void IdenticalBoxes()
    {
        var tg = new TG();
        var boxes = tg.InF("boxes", 1, 300, 4);
        var scores = tg.InF("scores", 1, 2, 300);
        tg.B.Output("selected", Nms(tg, boxes, scores, 600, 300, 0.5f, 0f, false));
        var data = new float[300 * 4];
        for (int i = 0; i < 300; i++)
        {
            (data[i * 4], data[i * 4 + 1], data[i * 4 + 2], data[i * 4 + 3]) = i % 3 == 0 ? (0f, 0f, 1f, 1f) : (0f, 0.6f, 1f, 1.6f);
        }

        Diff.Check("NonMaxSuppression", tg.B.Build(), new Step(1, 0, (boxes, data), (scores, Enumerable.Repeat(0.9f, 600).ToArray())), atol: 0, rtol: 0, what: "identical");
    }

    /// <summary>A symbolic batch, and runs with different counts through the same compiled model.</summary>
    [Fact]
    public void SymbolicBatch()
    {
        var tg = new TG();
        var boxes = tg.InF("boxes", TG.T, 256, 4);
        var scores = tg.InF("scores", TG.T, 7, 256);
        tg.B.Output("selected", Nms(tg, boxes, scores, TG.T * (7 * 64), 64, 0.5f, 0.1f, false));
        var graph = tg.B.Build();
        var steps = new List<Step>();
        for (int t = 1; t <= 3; t++)
        {
            steps.Add(new Step(t, 0, (boxes, Boxes(t * 256L, 10 + t)), (scores, Scores(t * 7L * 256, 20 + t))));
        }

        Diff.Check("NonMaxSuppression", graph, steps, atol: 0, rtol: 0, maxTokens: 4, what: "symbolic batch");
    }

    /// <summary>One batch and class with a symbolic box count ([S, 4] and [S]), the detection-head form.</summary>
    [Fact]
    public void SymbolicBoxCount()
    {
        var tg = new TG();
        var boxes = tg.InF("boxes", TG.T, 4);
        var scores = tg.InF("scores", TG.T);
        tg.B.Output("selected", Nms(tg, boxes, scores, TG.T, 100, 0.5f, 0.2f, false));
        var graph = tg.B.Build();
        Diff.Check("NonMaxSuppression", graph, [new Step(900, 0, (boxes, Boxes(900, 31)), (scores, Scores(900, 32))), new Step(77, 0, (boxes, Boxes(77, 33)), (scores, Scores(77, 34)))],
            atol: 0, rtol: 0, maxTokens: 1000, what: "symbolic boxes");
    }

    /// <summary>
    /// Consumers of the selection in the next GPU segment: the count binds their extent, and a top-k over the selected scores takes a
    /// symbolic k (the SSD post-processing pattern).
    /// </summary>
    [Fact]
    public void ConsumedOnTheGpu()
    {
        var tg = new TG();
        var boxes = tg.InF("boxes", 1, 2000, 4);
        var scores = tg.InF("scores", 1, 6, 2000);
        string countSymbol = tg.B.DynamicSymbol("nms", 6 * 200);
        var selected = tg.B.Emit1("NonMaxSuppression", [boxes, scores],
            Attributes.From(("count_symbol", countSymbol), ("max_per_class", 200L), ("iou_threshold", 0.5f), ("score_threshold", 0.1f)));
        var asFloat = tg.B.Emit1("Cast", [selected], Attributes.From(("type", "f32")));
        var classes = tg.B.Emit1("Cast", [tg.B.Emit1("Reshape", [tg.B.Slice(asFloat, 1, 1, 1)], Attributes.From(("shape", new long[] { 0 })))], Attributes.From(("type", "i32")));
        var boxIndex = tg.B.Emit1("Cast", [tg.B.Emit1("Reshape", [tg.B.Slice(asFloat, 1, 2, 1)], Attributes.From(("shape", new long[] { 0 })))], Attributes.From(("type", "i32")));
        var flatIndex = tg.B.Binary("Add", tg.B.Binary("Mul", classes, tg.B.Constant("n", DataType.I32, new Shape(1), BitConverter.GetBytes(2000))), boxIndex);
        var flatScores = tg.B.Reshape(scores, 12000);
        var picked = tg.B.Emit1("GatherAxis", [flatScores, flatIndex], Attributes.From(("axis", 0L)));
        var top = tg.B.Emit("TopK", [picked], Attributes.From(("k_symbol", $"min(50,{countSymbol})"), ("axis", 0L), ("largest", true)));
        tg.B.Output("selected", selected);
        tg.B.Output("top_scores", top[0]);
        tg.B.Output("top_index", top[1]);
        Diff.Check("NonMaxSuppression", tg.B.Build(), new Step(1, 0, (boxes, Boxes(2000, 41)), (scores, Scores(12000, 42, 20))), atol: 0, rtol: 0, what: "consumed");
    }
}
