#!/usr/bin/env bash
# Is an Ollama server running on the Windows host reachable from WSL? Prints the first URL that answers.
# On Windows: set OLLAMA_HOST=0.0.0.0:11434 (user environment variable), restart Ollama, and allow port 11434 through the Windows firewall for the WSL network.
for host in "$(ip route show default | awk '{print $3}')" "$(awk '/nameserver/{print $2}' /etc/resolv.conf | head -1)" host.docker.internal; do
  [ -z "$host" ] && continue
  if curl -s -m 3 -o /dev/null "http://$host:11434/api/tags"; then echo "http://$host:11434"; exit 0; fi
done
echo "no Windows Ollama reachable from WSL (see docs/NATIVE_SPEED_RUN.md)" >&2
exit 1
