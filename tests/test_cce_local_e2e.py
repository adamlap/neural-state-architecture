from experiments.cce_local_e2e import main


def test_cce_local_e2e(monkeypatch, tmp_path):
    output = tmp_path / "e2e.json"
    monkeypatch.setattr("sys.argv", ["cce_local_e2e", "--duration", "0", "--output", str(output)])
    assert main() == 0
    assert output.exists()
