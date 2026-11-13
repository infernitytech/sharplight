using System.Globalization;

namespace Sharplight;

/// <summary>Conversion of loosely typed attribute and metadata values to the type a caller asks for.</summary>
internal static class ValueConverter
{
    /// <summary>
    /// Converts <paramref name="value"/> to <typeparamref name="T"/>: values that already have the type pass through, arrays
    /// are converted element-wise, everything else goes through <see cref="Convert.ChangeType(object, Type, IFormatProvider)"/>
    /// with the invariant culture.
    /// </summary>
    /// <param name="value">The stored value.</param>
    /// <param name="subject">What the value is, for error messages (for example "Attribute").</param>
    /// <param name="key">The key the value was stored under, for error messages.</param>
    /// <exception cref="InvalidCastException">The value cannot be converted.</exception>
    public static T Convert<T>(object value, string subject, string key)
    {
        if (value is T typed)
        {
            return typed;
        }

        var target = typeof(T);
        if (target.IsArray && value is Array source)
        {
            var elementType = target.GetElementType()!;
            var converted = Array.CreateInstance(elementType, source.Length);
            for (int i = 0; i < source.Length; i++)
            {
                converted.SetValue(System.Convert.ChangeType(source.GetValue(i), elementType, CultureInfo.InvariantCulture), i);
            }

            return (T)(object)converted;
        }

        try
        {
            return (T)System.Convert.ChangeType(value, target, CultureInfo.InvariantCulture);
        }
        catch (Exception e)
        {
            throw new InvalidCastException($"{subject} '{key}' ({value.GetType().Name}) is not convertible to {target.Name}.", e);
        }
    }
}
