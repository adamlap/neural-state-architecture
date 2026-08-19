"""
nsa/runtime/inference/action_parser.py
======================================
Robust JSON Action Parser for Real LLM Generation Outputs.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


class ActionParser:
    """Extracts and validates structured action JSON from raw LLM text responses."""

    @classmethod
    def extract_action_json(cls, raw_text: str) -> Optional[Dict[str, Any]]:
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()

        # 1. Try direct JSON parsing
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass

        # 2. Extract JSON from Markdown code blocks (```json ... ``` or ``` ... ```)
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        # 3. Find outermost curly braces { ... }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                candidate = text[start : end + 1]
                data = json.loads(candidate)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

        # 4. Preserve raw_text for pattern analysis if JSON extraction failed
        return {"raw_text": text}

    @classmethod
    def sanitize_action_proposal(
        cls,
        parsed_data: Optional[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        default_fallback: str = "probe_service_config",
        strict_live: bool = False,
    ) -> Dict[str, Any]:
        valid_tool_names = {t["name"] for t in available_tools}

        if parsed_data:
            # Check proposed_action dict format
            if "proposed_action" in parsed_data and isinstance(parsed_data["proposed_action"], dict):
                action_name = parsed_data["proposed_action"].get("tool", "")
                params = parsed_data["proposed_action"].get("arguments", {})
            else:
                action_name = parsed_data.get("action", parsed_data.get("tool", ""))
                params = parsed_data.get("params", parsed_data.get("arguments", {}))

            thought = parsed_data.get("thought", parsed_data.get("reasoning", "LLM reasoning step"))
            try:
                confidence = float(parsed_data.get("confidence", parsed_data.get("epistemic_confidence", 0.80)))
            except Exception:
                confidence = 0.80

            # 1. Exact valid tool match
            if action_name in valid_tool_names:
                return {
                    "thought": str(thought),
                    "action": str(action_name),
                    "params": dict(params) if isinstance(params, dict) else {},
                    "confidence": max(0.0, min(1.0, confidence)),
                }

            # 2. Nested parameter tool lookup
            if isinstance(params, dict):
                nested_tool = params.get("tool", params.get("name", params.get("action", "")))
                if nested_tool in valid_tool_names:
                    return {
                        "thought": str(thought),
                        "action": str(nested_tool),
                        "params": dict(params),
                        "confidence": max(0.0, min(1.0, confidence)),
                    }
                for k in params:
                    if k in valid_tool_names:
                        return {
                            "thought": str(thought),
                            "action": str(k),
                            "params": dict(params),
                            "confidence": max(0.0, min(1.0, confidence)),
                        }

            # 3. Substring tool match in action name (e.g. "probe_storage_volume()")
            if isinstance(action_name, str):
                for tool in valid_tool_names:
                    if tool in action_name:
                        return {
                            "thought": str(thought),
                            "action": str(tool),
                            "params": dict(params) if isinstance(params, dict) else {},
                            "confidence": max(0.0, min(1.0, confidence)),
                        }

            # 4. Scan thought, reasoning, and raw text for valid tool mentions
            text_candidates = []
            if "thought" in parsed_data and isinstance(parsed_data["thought"], str):
                text_candidates.append(parsed_data["thought"])
            if "reasoning" in parsed_data and isinstance(parsed_data["reasoning"], str):
                text_candidates.append(parsed_data["reasoning"])
            if "raw_text" in parsed_data and isinstance(parsed_data["raw_text"], str):
                text_candidates.append(parsed_data["raw_text"])

            for text_to_scan in text_candidates:
                found_tools = []
                for tool_name in valid_tool_names:
                    matches = [m.start() for m in re.finditer(r'\b' + re.escape(tool_name) + r'\b', text_to_scan)]
                    for m in matches:
                        found_tools.append((m, tool_name))
                if found_tools:
                    found_tools.sort(key=lambda x: x[0], reverse=True)
                    selected_tool = found_tools[0][1]
                    return {
                        "thought": text_to_scan[:200].replace("\n", " ").strip(),
                        "action": selected_tool,
                        "params": {},
                        "confidence": max(0.0, min(1.0, confidence)),
                    }

            # 5. Check if action is empty/omitted/no-op keyword
            is_empty_or_noop = not action_name or str(action_name).strip() == "" or str(action_name).strip().lower() in {
                "none", "no_action", "no_action_needed", "null", "nil", "skip", "wait", "n/a", "na", "nothing", "done", "completed", "finish", "pass", "idle", "standby"
            }
            if is_empty_or_noop and default_fallback and default_fallback in valid_tool_names:
                return {
                    "thought": str(thought) if thought else "Model completed belief update; transitioning to verified remediation step.",
                    "action": str(default_fallback),
                    "params": {},
                    "confidence": max(0.0, min(1.0, confidence)),
                }

            # 6. If the model proposed a named action (even if hallucinated by unguided baseline), preserve it for environment execution
            if action_name and str(action_name).strip():
                return {
                    "thought": str(thought),
                    "action": str(action_name).strip(),
                    "params": dict(params) if isinstance(params, dict) else {},
                    "confidence": max(0.0, min(1.0, confidence)),
                }

        if strict_live and not default_fallback:
            raw_preview = str(parsed_data.get("raw_text", parsed_data)) if isinstance(parsed_data, dict) else str(parsed_data)
            raise ValueError(
                f"[STRICT LIVE INFERENCE FAILURE] Model did not propose a valid tool from {valid_tool_names}. "
                f"Raw output: {raw_preview}"
            )

        # Fallback to default
        fallback = default_fallback if (default_fallback and default_fallback in valid_tool_names) else (list(valid_tool_names)[0] if valid_tool_names else "probe_service_config")
        return {
            "thought": str(parsed_data.get("thought", "Fallback heuristic proposal based on available tools.")) if isinstance(parsed_data, dict) else "Fallback heuristic proposal.",
            "action": fallback,
            "params": {},
            "confidence": 0.50,
        }
