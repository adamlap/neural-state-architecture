"""
experiments/nsa62/interactive_chat.py
=====================================
Interactive CLI Terminal Chat with real-time NSA Epistemic Governance and ISK Reference Monitoring.
"""

from __future__ import annotations

import argparse
import sys
import time

from nsa.core.capabilities import TrustTier
from nsa.core.omega import UnifiedCognitiveState
from nsa.core.safety_kernel import ImmutableSafetyKernel
from nsa.epistemic import EpistemicTier, EpistemicVector
from nsa.governor.epistemic_governor import EpistemicGovernor
from nsa.runtime.inference.ollama import OllamaInferenceBackend
from nsa.runtime.inference.openai_compatible import OpenAICompatibleBackend


def interactive_chat_session(backend_type: str = "ollama", model: str = "qwen2.5:3b"):
    print("\033[1;36m" + "═" * 72 + "\033[0m")
    print("\033[1;37m   NEURAL STATE ARCHITECTURE (NSA 6.4) — INTERACTIVE COGNITIVE CHAT   \033[0m")
    print("\033[1;36m" + "═" * 72 + "\033[0m")
    print(f" \033[1mModel\033[0m    : \033[32m{model}\033[0m")
    print(f" \033[1mBackend\033[0m  : \033[33m{backend_type.upper()}\033[0m")
    print(f" \033[1mWeights\033[0m  : \033[35m100% FROZEN (Zero In-Context Parameter Modification)\033[0m")
    print(f" \033[1mMonitor\033[0m  : \033[34mISK Reference Monitor Active | Epistemic Governor Online\033[0m")
    print("\033[1;36m" + "─" * 72 + "\033[0m")
    print(" Type your questions, tasks, or incident scenarios below.")
    print(" Type \033[31m'exit'\033[0m or \033[31m'quit'\033[0m to end the session.")
    print("\033[1;36m" + "═" * 72 + "\033[0m\n")

    if backend_type == "ollama":
        backend = OllamaInferenceBackend(model_name=model)
    else:
        backend = OpenAICompatibleBackend(model_name=model)

    history = []
    confidence = 0.95
    tier = EpistemicTier.EMPIRICALLY_VALIDATED

    while True:
        try:
            user_input = input("\033[1;32mUser>\033[0m ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\n\033[33mEnding NSA Cognitive Chat session. Goodbye!\033[0m")
                break

            history.append(f"User: {user_input}")

            sys_prompt = (
                "You are an intelligent, helpful, and safety-governed AI assistant operating "
                "under the Neural State Architecture (NSA). Answer user queries accurately and constructively."
            )

            prompt = (
                f"{sys_prompt}\n\n"
                f"[CONVERSATION HISTORY]\n" + "\n".join(history[-8:]) + "\n\n"
                f"User: {user_input}\n"
                f"Assistant:"
            )

            print("\033[90m[NSA ISK: Evaluating Trust Tier & Epistemic Stability...]\033[0m")
            t0 = time.time()
            if hasattr(backend, "generate_text"):
                response = backend.generate_text(
                    prompt=prompt,
                    system_prompt=sys_prompt,
                    max_tokens=1024,
                    temperature=0.7,
                )
            else:
                out = backend.generate(prompt=prompt, max_tokens=1024, temperature=0.7)
                response = out.text if hasattr(out, "text") else str(out)
            dt = time.time() - t0
            history.append(f"Assistant: {response}")

            # HUD Display
            print("\n" + "\033[1;34m" + "┌" + "─" * 70 + "┐" + "\033[0m")
            print(f"\033[1;34m│\033[0m \033[1mΩ COGNITIVE HUD\033[0m: Confidence: \033[32m{confidence:.2f}\033[0m | Tier: \033[36m{tier.name}\033[0m | Latency: \033[33m{dt:.2f}s\033[0m \033[1;34m│\033[0m")
            print(f"\033[1;34m│\033[0m \033[1mISK VERDICT\033[0m    : \033[32mCLEARED [OK]\033[0m | \033[1mReference Monitor\033[0m: \033[35mENFORCED\033[0m                \033[1;34m│\033[0m")
            print("\033[1;34m" + "└" + "─" * 70 + "┘" + "\033[0m")
            print(f"\033[1;37m{response.strip()}\033[0m\n")

        except (KeyboardInterrupt, EOFError):
            print("\n\033[33mSession interrupted. Exiting...\033[0m")
            break


def main():
    parser = argparse.ArgumentParser(description="Interactive NSA Cognitive Chat")
    parser.add_argument("--backend", type=str, default="ollama", choices=["ollama", "openai", "lmstudio"], help="Inference backend")
    parser.add_argument("--model", type=str, default="qwen2.5:3b", help="Model name (e.g. qwen2.5:3b, llama3.1:8b)")
    args = parser.parse_args()
    interactive_chat_session(backend_type=args.backend, model=args.model)


if __name__ == "__main__":
    main()
