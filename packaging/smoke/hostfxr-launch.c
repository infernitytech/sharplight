/* Starts a self-contained .NET app through its libhostfxr with explicit paths, so that the app can run under
 * `ld.so --inhibit-cache` (the apphost finds its directory through /proc/self/exe, which then names the loader).
 * Usage: hostfxr-launch <app dir> <app name> [app arguments...] */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int (*main_startupinfo_fn)(int argc, const char **argv, const char *host_path, const char *dotnet_root, const char *app_path);

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        fprintf(stderr, "usage: %s <app dir> <app name> [arguments...]\n", argv[0]);
        return 2;
    }

    char hostfxr[4096], host[4096], app[4096];
    snprintf(hostfxr, sizeof hostfxr, "%s/libhostfxr.so", argv[1]);
    snprintf(host, sizeof host, "%s/%s", argv[1], argv[2]);
    snprintf(app, sizeof app, "%s/%s.dll", argv[1], argv[2]);
    void *library = dlopen(hostfxr, RTLD_NOW);
    main_startupinfo_fn start = library ? (main_startupinfo_fn)dlsym(library, "hostfxr_main_startupinfo") : NULL;
    if (start == NULL)
    {
        fprintf(stderr, "cannot load %s: %s\n", hostfxr, dlerror());
        return 2;
    }

    /* argv for hostfxr: the host path, then the app's own arguments. */
    argv[2] = host;
    return start(argc - 2, (const char **)argv + 2, host, argv[1], app);
}
