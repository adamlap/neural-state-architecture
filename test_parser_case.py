from nsa.runtime.inference.action_parser import ActionParser

tools = [
    {"name": "probe_service_config"},
    {"name": "staged_reload_config"},
    {"name": "promote_staged_cluster"},
]
data = {
    "thought": "The current belief entropy H(B_t) is 0.28 bits, which is below the threshold of 0.50 bits required for state mutations. Therefore, no immediate action is needed as the current diagnostic probes have already covered a significant portion of the identified hypotheses.",
    "action": "",
    "params": {},
    "confidence": 0.95,
}

res = ActionParser.sanitize_action_proposal(
    data,
    tools,
    default_fallback="staged_reload_config",
    strict_live=True,
)
print("Sanitized proposal result:", res)
assert res["action"] == "staged_reload_config"
print("SUCCESS: ActionParser handled empty string gracefully!")
