using System.Globalization;

namespace Sharplight.Convert;

/// <summary>A device named as <c>provider:index</c> (e.g. <c>vulkan:0</c>); the index defaults to 0.</summary>
internal readonly record struct DeviceSpec(string Provider, int Index)
{
    private const char Separator = ':';

    public static bool TryParse(string text, out DeviceSpec spec)
    {
        spec = default;
        var parts = text.Split(Separator);
        if (parts.Length > 2 || parts[0].Length == 0)
        {
            return false;
        }

        int index = 0;
        if (parts.Length == 2 && (!int.TryParse(parts[1], NumberStyles.None, CultureInfo.InvariantCulture, out index)))
        {
            return false;
        }

        spec = new DeviceSpec(parts[0], index);
        return true;
    }

    public override string ToString() => $"{Provider}{Separator}{Index}";
}
