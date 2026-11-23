# Logs every vLLM V1 scheduler step (time, decode rows, prompt rows, prompt requests) to $VLLM_STEP_LOG.<pid>.
import os, sys
_path = os.environ.get("VLLM_STEP_LOG")
if _path:
    import importlib.abc, importlib.util, time
    _out = None
    def _wrap(mod):
        S = mod.Scheduler
        orig = S.schedule
        def schedule(self, *a, **k):
            global _out
            o = orig(self, *a, **k)
            try:
                n = o.num_scheduled_tokens
                if n:
                    dec = sum(1 for v in n.values() if v == 1)
                    pre = [v for v in n.values() if v > 1]
                    if _out is None:
                        _out = open(f"{_path}.{os.getpid()}", "a", buffering=1)
                    _out.write(f"{time.perf_counter():.6f} {o.total_num_scheduled_tokens} {dec} {sum(pre)} {len(pre)}\n")
            except Exception as e:
                sys.stderr.write(f"steplog: {e}\n")
            return o
        S.schedule = schedule
    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "vllm.v1.core.sched.scheduler":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            sys.meta_path.insert(0, self)
            loader = spec.loader
            class L(importlib.abc.Loader):
                def create_module(self, s): return loader.create_module(s)
                def exec_module(self, m):
                    loader.exec_module(m); _wrap(m)
            spec.loader = L()
            return spec
    sys.meta_path.insert(0, _Finder())
