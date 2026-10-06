"""
nsa/server
==========
OpenAI and Ollama compatible API server for NSA-governed real-model cognitive runtime.
"""

from nsa.server.dashboard import DASHBOARD_HTML
from nsa.server.proxy import NSAHTTPHandler, NSAProxyRuntime, run_server

__all__ = [
    "DASHBOARD_HTML",
    "NSAHTTPHandler",
    "NSAProxyRuntime",
    "run_server",
]
