from nsa.runtime.inference.transformers import PyTorchTransformersBackend
from nsa.runtime.inference.base import BackendMode

backend = PyTorchTransformersBackend(model_name="Qwen/Qwen2.5-0.5B-Instruct", mode=BackendMode.CACHED)
out = backend.generate("""You are an autonomous cognitive agent. Propose a JSON action matching:
{"thought": "...", "action": "staged_reload_config", "params": {}}
Available tools: staged_reload_config, promote_staged_cluster, probe_service_config.

Propose your next action tool:""")
print("RAW GENERATION:\n", repr(out.text))
