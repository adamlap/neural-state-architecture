from nsa import NSA, OllamaBackend

agent = NSA(OllamaBackend("qwen2.5:3b"), initial_state={"goal": "be useful and safe"})
result = agent.run("Explain why persistent state can help an agent reason over time.")
print(result.text)
print(result.state)
