namespace Sharplight.Shaders.Lowering;

/// <summary>
/// Host-side source indices of a nearest Resize axis, computed exactly like the CPU provider (and ONNX Runtime): single
/// precision coordinate math with IEEE division. The GPU kernel looks the indices up instead of recomputing them, because
/// Vulkan division is not correctly rounded (2.5 ULP) and the drivers may rewrite residual-correction tricks, so exact
/// round-half ties (e.g. 8 -> 7 half_pixel: (3 + 0.5) / 0.875 - 0.5 = 3.5) would land on the other neighbour (#49, #67).
/// </summary>
internal static class ResizeNearestTable
{
    /// <summary>Source index (clamped to [0, lenIn)) for every output position of one axis.</summary>
    public static long[] Indices(int lenIn, int lenOut, string coord, string nearestMode)
    {
        var r = new long[lenOut];
        float scale = (float)lenOut / lenIn;
        for (int o = 0; o < lenOut; o++)
        {
            float x = coord switch
            {
                "align_corners" => lenOut == 1 ? 0f : (float)o * (lenIn - 1) / (lenOut - 1),
                "asymmetric" => o / scale,
                "pytorch_half_pixel" => lenOut > 1 ? (o + 0.5f) / scale - 0.5f : 0f,
                _ => (o + 0.5f) / scale - 0.5f,
            };
            long idx = nearestMode switch
            {
                "floor" => (long)MathF.Floor(x),
                "ceil" => (long)MathF.Ceiling(x),
                "round_prefer_ceil" => x == (long)x + 0.5f ? (long)MathF.Ceiling(x) : (long)MathF.Round(x, MidpointRounding.AwayFromZero),
                _ => x == (long)x + 0.5f ? (long)MathF.Floor(x) : (long)MathF.Round(x, MidpointRounding.AwayFromZero),
            };
            r[o] = Math.Clamp(idx, 0, lenIn - 1);
        }
        return r;
    }
}
