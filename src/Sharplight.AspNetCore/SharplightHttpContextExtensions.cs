using Microsoft.AspNetCore.Http;

namespace Sharplight.AspNetCore;

/// <summary>Access to the Sharplight request state from handlers and middleware.</summary>
public static class SharplightHttpContextExtensions
{
    extension(HttpContext http)
    {
        /// <summary>The context the <see cref="IModelEndpointResolver"/> resolved for this request, or null outside the Sharplight endpoints.</summary>
        /// <returns>The context, or null.</returns>
        public SharplightRequestContext? GetSharplightRequestContext() => SharplightRequestFeature.Of(http)?.Context;
    }
}
