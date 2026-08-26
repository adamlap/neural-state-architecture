import sys
from nsa.runtime.inference.ollama import OllamaInferenceBackend

b = OllamaInferenceBackend(model_name="qwen2.5:3b")
print("Resolved base_url:", b.base_url)
print("Resolved model_name:", b.model_name)
res = b.propose_action(
    system_context="You are an autonomous incident remediation agent.",
    task_instruction="Cluster latency spike detected. What is your next action?",
    available_tools=[
        {"name": "probe_service_config", "description": "Inspect config", "trust_tier": "T1"},
        {"name": "staged_reload_config", "description": "Reload config", "trust_tier": "T2"},
    ],
)
print("Proposal result:", res)
