namespace Sharplight.Vision;

/// <summary>Host-side post-processing shared by detection models: non-maximum suppression.</summary>
public static class BoxUtilities
{
    /// <summary>
    /// Greedy non-maximum suppression: visits boxes by descending score and keeps one unless it overlaps an already kept box (of the same class
    /// when <paramref name="classes"/> is given) by more than <paramref name="iouThreshold"/>.
    /// </summary>
    /// <param name="boxes">The candidate boxes.</param>
    /// <param name="scores">One score per box.</param>
    /// <param name="iouThreshold">Boxes overlapping a kept box by more than this intersection over union are dropped.</param>
    /// <param name="classes">One class per box (suppression only within a class), or <see langword="null"/> to suppress across classes.</param>
    /// <returns>The indices of the kept boxes, best score first.</returns>
    public static int[] NonMaxSuppression(IReadOnlyList<BoundingBox> boxes, IReadOnlyList<float> scores, float iouThreshold, IReadOnlyList<int>? classes = null)
    {
        ArgumentNullException.ThrowIfNull(boxes);
        ArgumentNullException.ThrowIfNull(scores);
        var order = Enumerable.Range(0, boxes.Count).OrderByDescending(i => scores[i]).ThenBy(i => i);
        var kept = new List<int>();
        foreach (int i in order)
        {
            bool suppressed = false;
            foreach (int k in kept)
            {
                if ((classes is null || classes[i] == classes[k]) && boxes[i].IntersectionOverUnion(boxes[k]) > iouThreshold)
                {
                    suppressed = true;
                    break;
                }
            }

            if (!suppressed)
            {
                kept.Add(i);
            }
        }

        return [.. kept];
    }
}
