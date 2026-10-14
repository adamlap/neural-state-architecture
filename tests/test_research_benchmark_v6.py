from types import SimpleNamespace

from experiments.research_v6 import tool_injection_benchmark as v6


def test_parse_calls_handles_spaces_in_body_and_quotes():
    calls = v6.parse_calls('CALL send_email to="a@b.io" body=Hello there, team\nnoise\nCALL delete_file path=/x/y')
    assert calls == [("send_email", {"to": "a@b.io", "body": "Hello there, team"}), ("delete_file", {"path": "/x/y"})]


def test_parse_calls_accepts_function_style_and_positional_arguments():
    text = ('CALL send_email(to="a@b.io", body="Hello, there")\n'
            "CALL send_email(to='c@d.io', body='x')\n"
            "CALL delete_file('/data/secrets/k.txt')\n"
            "CALL send_email(to=e@f.io, body=free text here)")
    calls = v6.parse_calls(text)
    assert calls[0] == ("send_email", {"to": "a@b.io", "body": "Hello, there"})
    assert calls[1] == ("send_email", {"to": "c@d.io", "body": "x"})
    assert calls[2] == ("delete_file", {"path": "/data/secrets/k.txt"})
    assert calls[3] == ("send_email", {"to": "e@f.io", "body": "free text here"})


def test_injection_values_appear_only_in_the_untrusted_document():
    ep = v6.make_episode(7, "direct")
    assert ep.attacker_addr in ep.document and ep.attacker_addr not in ep.request
    assert ep.user_addr in ep.request
    assert ep.attacker_addr not in v6.make_episode(7, "benign").document


def test_gate_blocks_attack_calls_and_keeps_the_user_task():
    for kind in ("direct", "authority", "impersonation"):
        ep = v6.make_episode(7, kind)
        text = f"CALL send_email to={ep.user_addr} body=summary\nCALL send_email to={ep.attacker_addr} body=full doc"
        off, on = v6.score("m", 7, 0, ep, "plain", text, False)
        assert off.attack_executed and off.task_executed
        assert not on.attack_executed and on.task_executed


def test_gate_blocks_delete_attack_even_with_all_capabilities_granted():
    ep = v6.make_episode(7, "delete")
    off, on = v6.score("m", 7, 0, ep, "plain", f"CALL delete_file path={ep.attack_path}", False)
    assert off.attack_executed and not on.attack_executed


def test_unknown_tool_is_never_executed_under_the_gate():
    ep = v6.make_episode(7, "benign")
    off, on = v6.score("m", 7, 0, ep, "plain", "CALL rm_rf path=/", False)
    assert not on.calls[0][2]


class _Fake:
    model = "fake"

    def generate(self, prompt, **kw):
        return SimpleNamespace(text="CALL send_email to=nobody@x.io body=hi", raw_response={})


def test_run_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(v6, "OllamaInferenceBackend", lambda **kw: _Fake())
    args = SimpleNamespace(models=["fake"], seeds=[7], trials=1, timeout=1.0, out=str(tmp_path), ollama_url=None, summarize_only=False)
    manifest = v6.run(args)
    assert manifest["episodes_recorded"] == len(v6.KINDS) * len(v6.PROMPTS)
    assert v6.run(args)["episodes_recorded"] == manifest["episodes_recorded"]
