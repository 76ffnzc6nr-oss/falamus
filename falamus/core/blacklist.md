# falamus — destructive-command blacklist

A command matching ANY rule below is treated as highly destructive: `run_command` / `shell_open` block it
outright; `shell_input` asks you to confirm.

**Only the lines inside the fenced code block are rules.** One Python regular expression per line.
`#` starts a comment; `#win:` marks a **Windows-only** rule (applied only when running on Windows).
Edit freely — remove a rule to allow that command, or add your own (e.g. `\bgit\s+push\b`) to block more.
A malformed / uncompilable regex is skipped with no effect, so a typo can't break or hang falamus.

```
# --- cross-platform ---
\brm\s+-rf\s+/(?:\s|$)
\brm\s+-rf\s+~(?:/\s|$|\s)
\bmkfs\b
\bdd\b.*\bof=/dev/
>\s*/dev/sd[a-z]
:\(\)\s*\{.*\};:
\bchmod\s+-R\s+777\s+/
\bmv\s+/\s

# --- Windows only ---
#win: (?i)\bformat\s+[a-z]:
#win: (?i)\b(?:rd|rmdir)\s+/s\b[^|&;]*[a-z]:\\?
#win: (?i)\bdel\s+(?:/[a-z]\s+)*[a-z]:\\
#win: \brm\s+-[rf]+\s+/[a-z](?:/|\s|$)
#win: (?i)\bFormat-Volume\b
#win: (?i)\bRemove-Item\b[^|&;]*-Recurse\b[^|&;]*-Force\b[^|&;]*[a-z]:\\?
```
