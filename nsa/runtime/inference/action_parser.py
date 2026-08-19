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
            confidence = float(parsed_data.get("confidence", parsed_data.get("epistemic_confidence", 0.80)))

            if action_name in valid_tool_names:
                return {
                    "thought": str(thought),
                    "action": str(action_name),
                    "params": dict(params) if isinstance(params, dict) else {},
                    "confidence": max(0.0, min(1.0, confidence)),
                }

            # If raw_text was stored, scan for tool names in reverse order of appearance
            if "raw_text" in parsed_data:
                text_to_scan = parsed_data["raw_text"]
                found_tools = []
                for tool_name in valid_tool_names:
                    matches = [m.start() for m in re.finditer(r'\b' + re.escape(tool_name) + r'\b', text_to_scan)]
                    for m in matches:
                        found_tools.append((m, tool_name))
                if found_tools:
                    # Pick the tool mentioned closest to the end (final conclusion)
                    found_tools.sort(key=lambda x: x[0], reverse=True)
                    selected_tool = found_tools[0][1]
                    return {
                        "thought": text_to_scan[:200].replace("\n", " ").strip(),
                        "action": selected_tool,
                        "params": {},
                        "confidence": 0.85,
                    }

        # Fallback to default
        return {
            "thought": "Fallback heuristic proposal based on available tools.",
            "action": default_fallback if default_fallback in valid_tool_names else list(valid_tool_names)[0],
            "params": {},
            "confidence": 0.50,
        }
