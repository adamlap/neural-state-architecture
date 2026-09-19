# NSA Neural Residency Architecture

NSA now has a first-class neural virtual memory subsystem. The logical model can be larger than the physical accelerator/host memory footprint: weights are addressable as regions and their physical placement is managed separately from the cognitive/control state.

The first target models are the local Qwen2.5 1.5B and 3B models already used by the NSA experiments.

## Selective storage vs selective computation

The residency layer answers: Which weights should physically exist in the fast memory tier now?

The model layer answers: Which weights should be executed?

These are deliberately separate decisions.

## Current implementation

nsa.residency provides NeuralRegion, ResidencyPolicy, HeuristicResidencyPredictor, ResidencyCache and NeuralResidencyManager.

SelectiveStorageTransformersBackend provides a Transformers/Accelerate backend using an empty model skeleton and checkpoint-backed dispatch. Accelerate can use disk-backed overflow, so the full model does not need to be duplicated in host RAM.

NSA now has an active predictive control path. At decoder boundaries the learned/heuristic predictor plans the next regions. `AccelerateDiskPrefetcher` warms the NVMe-backed safetensor pages asynchronously, while Accelerate remains the owner of actual parameter materialization. This is deliberately page-cache prefetch rather than direct mutation of Accelerate hook state; the model is only marked resident when a backend actually performs the transfer.

## Running the targets

Install the ml-residency extra, then point SelectiveStorageTransformersBackend at a local Qwen2.5-1.5B-Instruct or Qwen2.5-3B-Instruct checkpoint directory. Cached mode does not silently download missing weights.

## Development phases

1. Foundation — region model, policy, predictor, cache, telemetry.
2. Disk residency — empty-model + disk-backed checkpoint execution.
3. Active residency — predictive page-cache prefetch and retention hooks around decoder regions.
4. State coupling — use NSA cognitive state instead of prompt heuristics.
5. Learned residency — train the predictor from region transition traces.
6. MoE specialization — experts become independently resident neural regions.
7. Evaluation — compare peak VRAM/RAM, transfer bandwidth, latency, tokens/s and output quality.

Architectural invariant: Residency can change where a neural region lives, but it cannot change NSA authority, policy, or safety decisions.
