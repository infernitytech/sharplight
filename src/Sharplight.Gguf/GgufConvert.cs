using System.Globalization;

namespace Sharplight.Gguf;

/// <summary>Lenient numeric conversion of GGUF metadata values (the file's integer width rarely matches what callers ask for).</summary>
internal static class GgufConvert
{
    /// <summary>
    /// Converts <paramref name="o"/> to <typeparamref name="T"/>: identical types pass through, numeric primitives convert,
    /// arrays convert element-wise. Strings never convert to or from other types.
    /// </summary>
    /// <returns>False (with a default value) when the value is not convertible or overflows.</returns>
    public static bool TryConvert<T>(object o, out T? value)
    {
        if (o is T exact)
        {
            value = exact;
            return true;
        }

        var target = typeof(T);
        try
        {
            if (target.IsArray)
            {
                return TryConvertArray(o, target.GetElementType()!, out value);
            }

            if (TryConvertTo(target, o, out var converted))
            {
                value = (T)converted!;
                return true;
            }
        }
        catch (Exception e) when (e is InvalidCastException or OverflowException or FormatException)
        {
            // Not convertible: reported through the return value below.
        }

        value = default;
        return false;
    }

    private static bool TryConvertArray<T>(object o, Type elementType, out T? value)
    {
        if (o is not Array source)
        {
            value = default;
            return false;
        }

        var destination = Array.CreateInstance(elementType, source.Length);
        for (int i = 0; i < source.Length; i++)
        {
            if (!TryConvertTo(elementType, source.GetValue(i)!, out var converted))
            {
                value = default;
                return false;
            }

            destination.SetValue(converted, i);
        }

        value = (T)(object)destination;
        return true;
    }

    private static bool TryConvertTo(Type target, object v, out object? result)
    {
        result = null;
        if (target.IsInstanceOfType(v) || target == typeof(object))
        {
            result = v;
            return true;
        }

        if (v is string || target == typeof(string) || v is Array)
        {
            return false;
        }

        if (target.IsPrimitive || target == typeof(decimal))
        {
            try
            {
                result = Convert.ChangeType(v, target, CultureInfo.InvariantCulture);
                return true;
            }
            catch (Exception e) when (e is InvalidCastException or OverflowException)
            {
                return false;
            }
        }

        return false;
    }
}
