# falamus prompt fragments (cloud set) — runtime notes, terse (`{…}` placeholders filled by code)

## win_env
<!-- scope: Windows-only note -->
(On Windows, run_command runs via Git-Bash: use Unix commands and forward-slash paths, not cmd/PowerShell verbs or backslashes.)

## workspace
<!-- scope: main's staging→project note ({work}, {workdir}) -->
Sub-agent shared workspace (staging): {work}
Project directory (final destination): {workdir}
Sub-agents write files into the workspace and report them; after verifying, move the final files into the project dir with run_command mv/cp (don't re-type contents).

## turn_reminder
<!-- scope: per-turn re-anchor for main ({tools}) — blank for cloud (a capable model needs no re-anchor) -->

## output_file_one
<!-- scope: single-deliverable injection ({name}) -->
Your deliverable must be the file `{name}` — write it with the `<<<FILE:{name}>>>` header. If you re-delegate, pass the same output_name down.

## output_file_many
<!-- scope: multi-deliverable injection ({listed}) -->
Your deliverables are {listed} — write each with its own `<<<FILE:name>>>` header. Pass the same names down if you re-delegate.

## depth_budget
<!-- scope: remaining spawn levels ({n}) -->
You may delegate {n} more level(s) of sub-agents; at 0 you are a leaf — do the work yourself.

## iter_budget
<!-- scope: sub-agent tool-call cap ({n}) -->
Tool-call budget: up to {n} calls for this task; hitting the cap force-stops you.

## iter_budget_main
<!-- scope: main is not capped ({n}) -->
You (main) have no tool-call cap; sub-agents get {n} each.

## solo
<!-- scope: main works alone (max_depth 0) -->
SOLO: no sub-agents this run (no spawn tool). Do every step yourself and write outputs directly into the project directory; ignore any "delegate to sub-agents" guidance in the rules.

## turn_reminder_solo
<!-- scope: per-turn re-anchor for solo main ({tools}) — blank for cloud -->

## rules_header
<!-- scope: intro to falamus.md -->
--- This project's long-term rules (falamus.md); follow them ---

## summary
<!-- scope: /exit progress-summary prompt -->
Summarize the conversation below into a concise progress note: what was accomplished, key results/decisions, unfinished items, important files. Short bullets. Reply in the conversation's language.
