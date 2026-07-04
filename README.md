# falamus

An agent framework for LLMs (llama.cpp / ollama / …) with a set of built-in tools and MCP support.

## Install
```
pip install .            # from source (Python 3.9+)
falamus --set-provider   # choose the model source (see Configuration)
falamus                  # run in a working directory
```

## Built-in tools
- **Files** — `read_file` / `write_file` / `append_file` / `edit_file` / `list_dir`
- **Shell** — `run_command` (one-off), and an opt-in persistent interactive shell
  (`shell_open` / `shell_input` / `shell_close`; see Safety)
- **Images** — `view_image`
- **Coordination** — `spawn_subagent` (delegate a step), `deliver` (return a result),
  `memo` (private notes)

## Use external MCP servers (MCP client)
Add any MCP server; its tools appear in the agent's toolbox as `<server>__<tool>`:
```
falamus --add-mcp <name> -- <command>    # e.g.  --add-mcp time -- mcp-server-time
falamus --list-mcp
falamus --remove-mcp <name>
```
Stored in `~/.config/falamus/mcp_servers.json` (standard `mcpServers` shape). An
external tool call is confirmed each time.

## Configuration
```
falamus --set-provider   # model source: a LOCAL server (llama.cpp/ollama) or a CLOUD backend
falamus --config         # interactive settings menu
```

## Run as an MCP server
```
falamus --mcp     # MCP over stdio (the client spawns it — no port, no daemon)
```
- One connection = one session. The caller may tune **safe** params (depth, iteration
  caps, thinking) but **not** capability/security ones (shell, allowed paths, working
  directory, model) — those are fixed by your local config.
- Dangerous actions are routed to the caller for confirmation (or auto-approved within
  the fixed config); the destructive blacklist always applies; every tool call is
  written to an audit log.
- For a remote caller, wrap it in SSH (stdio carries across machines — still no port).

## Safety
falamus is **not sandboxed**: it runs shell commands and edits files with your own
privileges. The destructive-command blacklist, confirmations and path-allowlist are
guardrails, not a sandbox — run it in a **trusted directory** on a **trusted task**.
- The **persistent interactive shell is off by default** (POSIX only). On Windows it is
  unavailable and stays silently off; enabling it there just prints a warning.
- Exposing an **MCP server** lets the caller read any file you can — expose it only to a **trusted** caller.

## License
MIT — see [`LICENSE`](LICENSE).
