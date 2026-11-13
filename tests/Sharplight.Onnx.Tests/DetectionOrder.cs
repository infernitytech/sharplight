namespace Sharplight.Onnx.Tests;

/// <summary>
/// Detection outputs are sorted by score; two detections whose scores differ by float noise (1e-6 on a 0.06 score) may come out
/// in either order on another backend, which a row-by-row comparison reports as large box errors. This aligns the reference
/// to the actual order, only swapping rows within groups of tied scores, so every other difference is still caught.
/// </summary>
internal static class DetectionOrder
{
    /// <summary>Relative score difference under which two detections count as tied.</summary>
    public const double TieTolerance = 1e-4;

    /// <summary>
    /// Returns, for each actual row, the reference row it corresponds to: the same row unless that row's box differs and a
    /// later reference row with a tied score matches instead.
    /// </summary>
    /// <param name="actualBoxes">Actual boxes, <paramref name="width"/> values per row.</param>
    /// <param name="expectedBoxes">Reference boxes in reference order.</param>
    /// <param name="expectedScores">Reference scores, one per row.</param>
    /// <param name="width">Values per box row.</param>
    /// <param name="boxTolerance">Largest absolute coordinate difference of matching boxes.</param>
    public static int[] AlignNearTies(float[] actualBoxes, float[] expectedBoxes, float[] expectedScores, int width, double boxTolerance)
    {
        int rows = expectedScores.Length;
        var order = Enumerable.Range(0, rows).ToArray();
        if (actualBoxes.Length != expectedBoxes.Length || expectedBoxes.Length != rows * width)
        {
            return order;   // different detection counts: nothing to align, the comparison reports it
        }

        for (int i = 0; i < rows; i++)
        {
            if (Matches(actualBoxes, i, expectedBoxes, order[i], width, boxTolerance))
            {
                continue;
            }

            double score = expectedScores[order[i]];
            for (int j = i + 1; j < rows && Tied(expectedScores[order[j]], score); j++)
            {
                if (Matches(actualBoxes, i, expectedBoxes, order[j], width, boxTolerance))
                {
                    (order[i], order[j]) = (order[j], order[i]);
                    break;
                }
            }
        }

        return order;
    }

    /// <summary>Reorders the rows of <paramref name="values"/> (any output with one row per detection) by <paramref name="order"/>.</summary>
    public static float[] Reorder(float[] values, int[] order)
    {
        if (order.Length == 0 || values.Length % order.Length != 0)
        {
            return values;
        }

        int width = values.Length / order.Length;
        var result = new float[values.Length];
        for (int i = 0; i < order.Length; i++)
        {
            Array.Copy(values, order[i] * width, result, i * width, width);
        }

        return result;
    }

    private static bool Tied(double a, double b) => Math.Abs(a - b) <= TieTolerance * Math.Max(Math.Abs(a), Math.Abs(b));

    private static bool Matches(float[] actual, int row, float[] expected, int expectedRow, int width, double tolerance)
    {
        for (int k = 0; k < width; k++)
        {
            if (Math.Abs(actual[(row * width) + k] - expected[(expectedRow * width) + k]) > tolerance)
            {
                return false;
            }
        }

        return true;
    }
}
