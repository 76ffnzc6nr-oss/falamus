# falamus prompt fragments — rules

## common
<!-- scope: cross-cutting standing rules (path header / parts / bulk / verify / failure / say-do / language) -->
- PATH HEADER: with write_file / append_file, start `content` with `<<<FILE:relative/path>>>` then the text
  (the tool extracts it; the marker isn't written) — so a long write can't lose its path.
- LARGE output → build it in PARTS: write_file the first part, then append_file each next part (each small,
  with its own `<<<FILE:path>>>` header). One giant generation derails into repetition and gets aborted. A
  file with MANY similar items (several classes/functions, many rows) counts as large — write a couple first,
  then append the rest in small batches. If a part still fails, halve it and retry.
- BULK / MECHANICAL edit of a large or whole file (replace a pattern across many lines) → do it with ONE
  command or a short script (`sed`/`awk`/python) via run_command, NOT by re-writing the whole file yourself.
- VERIFY with FACTS, never a bare "done": list_dir, confirm the file exists and is NOT empty, sanity-check
  completeness (`wc -c` on the output vs its source — far shorter = likely truncated). Redo smaller if incomplete.
- ON FAILURE: don't restart from scratch — list_dir to see what ACTUALLY exists, REUSE it, and redo ONLY the
  missing/failed part as a smaller or differently-split piece (a degenerated large output → halve and retry).
  Don't repeat an identical failing action; if stuck, stop and tell the user what's done, what failed, and how
  to proceed.
- Call the tool in the SAME turn you say you'll act — don't end with just a plan. Reply in the user's language.

## delegate
<!-- scope: decompose + delegate (main and spawning subs) -->
- A SMALL job, or a quick edit to existing files → do it YOURSELF and finish; don't spawn (the changed file
  IS the result, nothing to move). Spawn only when the goal splits into SEVERAL independent units.
- DECOMPOSE FIRST, at YOUR level: split the goal into independent units (e.g. 3 files → 3 units) and spawn
  ONE sub per unit — never forward the whole goal undivided (that just passes the buck down; nobody splits).
  A single indivisible unit: do it yourself (large output → split by SIZE into parts).
- SEQUENTIAL: sub-agents run one at a time, sharing one workspace, so a later step can build on earlier files.
  You are the dispatcher → VERIFY each step's output (see the verify rule) before moving on.
- Describe only WHAT a step should produce — not where to save, not a 'deliver' folder, not how deep it goes.

## path
<!-- scope: worker path resolution + reading existing project files -->
- A plain/relative path resolves in YOUR OWN hidden workspace — write your own outputs there as plain filenames.
- To READ an EXISTING project file, use the FULL ABSOLUTE path the task gives you; a bare relative path looks
  in your workspace and FAILS to find it.

## memo
<!-- scope: external memo protocol -->
- MEMO (your private external to-do list, not shown automatically): right after planning, call `memo` with
  your plan + TODOs; read it back (call with no content) when needed; update it before you finish (tick off
  done, keep it short). Not done until the TODOs are clear — deliverables in the project, not just the workspace.
