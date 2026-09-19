def test_selective_device_map_keeps_hot_edges_and_offloads_middle():
    from nsa.runtime.inference.resident_transformers import SelectiveStorageTransformersBackend

    backend = SelectiveStorageTransformersBackend(mode="mock", device="cpu", hot_layers=1, warm_layers=1)
    mapping = backend._selective_device_map(6)
    assert mapping["model.layers.0"] == "cpu"
    assert mapping["model.layers.5"] == "cpu"
    assert mapping["model.layers.1"] == "cpu"
    assert mapping["model.layers.4"] == "cpu"
    assert mapping["model.layers.2"] == "disk"
    assert mapping["model.layers.3"] == "disk"
