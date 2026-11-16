namespace Sharplight.Vision.Tests;

/// <summary>Patches of an input whose side is not a multiple of the patch size (SigLIP so400m: 384 px, 14 px patches).</summary>
public class PatchifyTests
{
    [Fact]
    public void The_remainder_at_the_right_and_bottom_edge_is_dropped()
    {
        // 5 x 4 pixels, patches of 2: a 2 x 2 grid; column 4 is dropped.
        var planes = new float[3 * 5 * 4];
        for (int i = 0; i < planes.Length; i++)
        {
            planes[i] = i;
        }

        var patches = new Sharplight.Imaging.ProcessedImage(planes, 5, 4, 5, 4, 1, 1, 0, 0).Patchify(2);
        Assert.Equal(2 * 2 * 3 * 2 * 2, patches.Length);
        Assert.Equal([0f, 1f, 5f, 6f], patches[..4]);                       // patch (0, 0), red plane
        Assert.Equal([2f, 3f, 7f, 8f], patches[(3 * 2 * 2)..((3 * 2 * 2) + 4)]);   // patch (0, 1), red plane
    }

    [Fact]
    public void A_patch_larger_than_the_image_is_rejected()
        => Assert.Throws<ArgumentException>(() => new Sharplight.Imaging.ProcessedImage(new float[3 * 4], 2, 2, 2, 2, 1, 1, 0, 0).Patchify(3));
}
