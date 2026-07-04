# falamus prompt fragments (cloud set) — tool descriptions, terse (only falamus-specific contracts kept)

## read_file
<!-- scope: read_file tool description -->
Read part of a text file. Requires 'offset' (start char) and 'limit' (chars; -1 = to end). Whole file: read_file(path, 0, -1). The result header names the slice (chars X-Y of TOTAL).

## write_file
<!-- scope: write_file tool description -->
Write (overwrite) a file. `content` MUST start with a path header: `<<<FILE:relative/path>>>` + the text (the marker sets the path and is not written to the file).

## append_file
<!-- scope: append_file tool description -->
Append to a file (same `<<<FILE:path>>>` header convention as write_file).

## edit_file
<!-- scope: edit_file tool description -->
Replace a uniquely-occurring piece of old text in a file with new text.

## list_dir
<!-- scope: list_dir tool description -->
List a directory's files and subdirectories.

## view_image
<!-- scope: view_image tool description -->
Load a local image so you can see it (png/jpg/gif/webp/bmp).

## run_command
<!-- scope: run_command tool description -->
Run a single shell command in the working directory; returns output and exit code.

## spawn_subagent
<!-- scope: spawn_subagent tool description -->
Delegate one step (with its own context) to a sub-agent. Sub-agents run one at a time, write files into the shared workspace, and report a summary. Pass output_name for the deliverable filename(s) (comma-separate multiple) so you collect the right ones. Describe WHAT to do, not where to save.

## memo
<!-- scope: memo tool description -->
Private scratchpad kept outside the conversation. No content = read; with `content` = overwrite. Each agent has its own.

## deliver
<!-- scope: deliver tool description -->
Report this task's result (summary); produced files are already in the shared workspace. Report any path written outside it via 'paths'. Call when done, then stop.

## shell_open
<!-- scope: shell_open tool description -->
Open a PERSISTENT interactive shell to drive a program across several exchanges (a REPL, an installer that asks questions, a program that reads stdin). For one-off commands, use run_command. Pass 'command' and a short 'note'; returns a handle (pid) + first output. The session lasts only THIS turn — close it before you reply.

## shell_input
<!-- scope: shell_input tool description -->
Send one line to a shell session (by 'handle') and read the response; omit 'input' to just poll for more output. Returns when the program goes quiet (awaiting input) or ends.

## shell_close
<!-- scope: shell_close tool description -->
Close a shell session (by 'handle'), terminating it and its children.
