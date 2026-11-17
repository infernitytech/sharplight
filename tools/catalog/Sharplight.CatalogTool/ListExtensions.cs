namespace Sharplight.CatalogTool;

/// <summary>Small list helpers.</summary>
internal static class ListExtensions
{
    extension<T>(List<T> list) where T : class
    {
        /// <summary>Adds <paramref name="item"/> unless it is null.</summary>
        public void AddIfNotNull(T? item)
        {
            if (item is not null)
            {
                list.Add(item);
            }
        }
    }
}
