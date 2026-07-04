# falamus prompt fragments — tool descriptions

## read_file
Read a text file — you MUST give the range: 'offset' (start char; 0 = beginning) and 'limit' (chars; -1 = the WHOLE file). Whole file: read_file(path, 0, -1). For a LARGE file, read in chunks (~{read_chunk} chars) and keep the next offset in your memo — the result header names the slice (chars X-Y of TOTAL), so your next offset is Y.

## write_file
Write content to a file (overwrites; creates parent dirs). Start `content` with the path header `<<<FILE:relative/path>>>` then the text (the marker isn't written to the file). For a large file, write_file the first part then append_file the rest.

## append_file
Append content to the END of a file (creates it if missing). Start `content` with `<<<FILE:relative/path>>>` then the part. Use this to build a large file in parts.

## edit_file
Replace a uniquely-occurring piece of old text in a file with new text.

## list_dir
List the files and subdirectories in a directory.

## view_image
Load a local image file (png/jpg/gif/webp/bmp) so you can see its contents.

## run_command
Run a single shell command in the working directory; returns output and exit code.

## spawn_subagent
Hand ONE step of your plan (heavy work that produces files) to a sub-agent with its own context — sub-agents run sequentially, produce files in the shared workspace, and report a summary. Describe only WHAT the step produces (not where to save, no folder). When it produces file(s), pass output_name — the deliverable filename(s), comma-separated for multiple (e.g. `a.md, b.md`), with clear DISTINCT names so they don't overwrite each other and you collect exactly those. Give a large output its own sub-agent.

## memo
Your private to-do list / scratchpad, kept OUTSIDE the conversation. Call with NO content to READ it; call with `content` to OVERWRITE (tick off done, add new, keep it short). Review it at the start of your work, update it before you finish. Each agent has its own.

## deliver
Report the task result: give a summary (produced files are already in your shared workspace). If you wrote anything outside the workspace, report its absolute path via 'paths'. Call this when done, then stop.

## shell_open
Open a PERSISTENT interactive shell and keep it alive to interact with a program back-and-forth — a REPL, an installer that asks questions, a program that reads stdin and replies on stdout. Use ONLY when you must react to output across several exchanges; for a one-off command that just runs and finishes, use run_command. Pass 'command' to launch (e.g. python3 game.py) and a short 'note'. Returns a handle (pid) + the first output; drive it with shell_input and end it with shell_close. The session lasts only THIS turn — finish and close it before you reply.

## shell_input
Send one line of input to a shell session (by 'handle') and read back what the program prints. Omit 'input' to just poll for more output. The read returns once the program goes quiet (waiting for your next input) or ends. If it ended, stop (shell_close is unnecessary).

## shell_close
Close a shell session (by 'handle'), terminating the program and its children. Always close a session once done.
