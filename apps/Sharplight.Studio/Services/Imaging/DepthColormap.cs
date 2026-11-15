using Sharplight.Vision;

namespace Sharplight.Studio.Services.Imaging;

/// <summary>Colors of a depth map: the normalized values (0 = the smallest, 1 = the largest) as grayscale or through a color ramp.</summary>
internal static class DepthColormap
{
    /// <summary>Grayscale.</summary>
    public const string Gray = "gray";

    /// <summary>The turbo ramp (blue through green to red).</summary>
    public const string Turbo = "turbo";

    /// <summary>The viridis ramp (purple through green to yellow).</summary>
    public const string Viridis = "viridis";

    /// <summary>The inferno ramp (black through red to yellow).</summary>
    public const string Inferno = "inferno";

    private static readonly (double Position, byte R, byte G, byte B)[] s_turbo =
    [
        (0.00, 48, 18, 59), (0.13, 70, 107, 227), (0.25, 40, 187, 236), (0.38, 49, 241, 153), (0.50, 164, 252, 60),
        (0.63, 237, 207, 58), (0.75, 251, 128, 34), (0.88, 200, 42, 5), (1.00, 122, 4, 3),
    ];

    private static readonly (double Position, byte R, byte G, byte B)[] s_viridis =
    [
        (0.00, 68, 1, 84), (0.25, 59, 82, 139), (0.50, 33, 145, 140), (0.75, 94, 201, 98), (1.00, 253, 231, 37),
    ];

    private static readonly (double Position, byte R, byte G, byte B)[] s_inferno =
    [
        (0.00, 0, 0, 4), (0.25, 87, 16, 110), (0.50, 188, 55, 84), (0.75, 249, 142, 9), (1.00, 252, 255, 164),
    ];

    /// <summary>The names of the color maps.</summary>
    public static IReadOnlyList<string> Names { get; } = [Gray, Turbo, Viridis, Inferno];

    /// <summary>Renders the depth map as an RGB image.</summary>
    /// <param name="depth">The depth map.</param>
    /// <param name="map">A name of <see cref="Names"/>; unknown names give grayscale.</param>
    /// <param name="invert">Whether the near and far ends are swapped.</param>
    public static ImageInput Render(DepthMap depth, string map, bool invert = false)
    {
        var gray = depth.ToGrayscale();
        var ramp = map switch
        {
            Turbo => s_turbo,
            Viridis => s_viridis,
            Inferno => s_inferno,
            _ => null,
        };
        var rgb = new byte[gray.Length * 3];
        for (int i = 0; i < gray.Length; i++)
        {
            byte value = invert ? (byte)(255 - gray[i]) : gray[i];
            if (ramp is null)
            {
                rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = value;
            }
            else
            {
                var (r, g, b) = Sample(ramp, value / 255.0);
                rgb[3 * i] = r;
                rgb[3 * i + 1] = g;
                rgb[3 * i + 2] = b;
            }
        }

        return new ImageInput(depth.Width, depth.Height, rgb);
    }

    private static (byte R, byte G, byte B) Sample((double Position, byte R, byte G, byte B)[] ramp, double t)
    {
        for (int i = 1; i < ramp.Length; i++)
        {
            if (t <= ramp[i].Position)
            {
                var (p0, r0, g0, b0) = ramp[i - 1];
                var (p1, r1, g1, b1) = ramp[i];
                double f = (t - p0) / (p1 - p0);
                return ((byte)(r0 + (r1 - r0) * f + 0.5), (byte)(g0 + (g1 - g0) * f + 0.5), (byte)(b0 + (b1 - b0) * f + 0.5));
            }
        }

        var last = ramp[^1];
        return (last.R, last.G, last.B);
    }
}
