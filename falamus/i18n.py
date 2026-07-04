"""Minimal i18n (default English).

t(key, **kw) returns a string for the current language; falls back to English, then to the key itself.
set_lang("en"|"zh") switches language.

i18n POLICY: only UI-chrome strings shown to the *user* (status, commands, hints, prompts) live here
and switch with /lang. Strings sent to the *model* (tool descriptions, tool results, agent prompts,
failure/circuit-breaker messages) stay ENGLISH regardless of UI language — the model reads English, and
the reply language follows the user's input, not the UI setting.
"""

from __future__ import annotations

_LANG = "en"

_MESSAGES: dict[str, dict[str, str]] = {
    "en": {
        "connecting": "Connecting to {url} …",
        "session_line": "session: {sid}   workdir: {wd}",
        "resumed": "Resumed session {sid}",
        "resume_tail": "── context restored to the model; recent messages ──",
        "config_restart_note": "Note: config changes take effect after restart or /reset (only lang applies live).",
        "welcome": "Type a message to chat. /help for commands, /exit to quit.",
        "prompt": "you ▷ ",
        "agent_reply": "main ◀ {out}",
        "bye": "Bye.",
        "unknown_cmd": "Unknown command: {cmd} (try /help)",
        "tools_label": "Main agent tools: {tools}",
        "kill_none": "No persistent shell sessions are open.",
        "kill_list": "Open shell sessions (pid  [agent]  note):",
        "kill_usage": "Use /kill <pid> to force-close one (only sessions falamus started).",
        "compacted": "Context compacted.",
        "no_compact": "Auto-compact is disabled.",
        "new_session": "Started a new session.",
        "no_sessions": "(no sessions)",
        "saved_config": "Config saved.",
        "lang_set": "Language set to: {lang}",
        "lang_usage": "Usage: /lang <en|zh>",
        "resume_usage": "Usage: /resume <sid>",
        "helper_loaded": "ok falamus.md loaded ({n} chars) — injected into the main agent's rules, effective from the first message",
        "no_tools_chat": "! Plain chat mode — no tools. No falamus.md in this folder (none will be created).",
        "no_tools_chat_rules": "! Plain chat mode — no tools. falamus.md found ({n} chars) — loaded as rules (plain chat won't create or modify it).",
        "ollama_pick_model": "ollama: no model is loaded — choose one to run:",
        "ollama_pick_prompt": "  model [0-{n}, default 0]: ",
        "loading_model": "Loading model {model} into ollama (first load can take a while) …",
        "confirm": "  !  {reason}\n  Allow? [y/N]: ",
        # workdir
        "ask_workdir": "No workdir set. Enter a path to create/use as your workspace:\n> ",
        "workdir_missing": "Workdir does not exist: {wd}",
        "ask_create": "Create it? [y/N]: ",
        "workdir_in_program": "Workdir cannot be inside the helper program directory. Choose another.",
        "workdir_empty": "Path cannot be empty.",
        "workdir_set": "Workdir set: {wd}",
        "created_workdir": "Created: {wd}",
        "cd_done": "Switched workdir: {wd}",
        "cd_bad": "Not a directory: {arg}",
        "bar_hint": "/help for commands",
        "working": "Working…",
        "streaming": "Outputting…",
        "standby": "Idle",
        "esc_interrupt": "esc to interrupt",
        "busy_note": "(busy — press ESC to interrupt)",
        "confirm_hint": "type y / n below",
        "interrupted": "[interrupted]",
        "ask_summary": "Update conversation summary to falamus.md before exit?",
        "saving_progress": "Saving conversation summary to falamus.md …",
        "progress_saved": "Progress saved to falamus.md",
        "dev_on": "Developer mode ON — confirmations skipped (destructive cmds still ask)",
        "dev_off": "Developer mode OFF",
        "ctx_label": "ctx",
        "ctx_warn": "! near auto-compact",
        "server_offline": "[server offline] Cannot reach the model server — check it is running.",
        "server_unreachable": (
            "Cannot reach the model server at {url}.\n"
            "  - Is the server running?  (llama.cpp: llama-server …   |   ollama: ollama serve)\n"
            "  - Set the address:  falamus --base-url http://HOST:PORT"
            "   (or in the TUI: /config base_url http://HOST:PORT)"
        ),
        "server_down": "! SERVER OFFLINE",
        "compacting": "Compacting context…",
        "idle": "* idle",
        "active_done": "active:{a} done:{d}",
        "help": """Commands:
  /help              show this help
  /version           show the version
  /b                 jump to the latest output & resume auto-scroll (= Ctrl-G)
  /tools             list main-agent tools
  /kill [pid]        list open shell sessions / force-close one (--persistent-interactive-shell)
  /config            show current config (config.ini)
  /compact           compact context now
  /reset             start a new session
  /sessions          list sessions in this workdir
  /resume <sid>      resume a session
  /cd <path>         switch workdir (auto-saved to config.ini)
  /lang <en|zh>      switch language
  /dev               toggle developer mode (skip confirmations)
  /save              save config
  /exit              quit

Keys:
  Tab                switch focus: output <-> input (focus the OUTPUT to scroll it)
  Up/Down PgUp/PgDn  scroll the output (after Tab); scroll to the bottom resumes auto-follow
  Ctrl-G             jump to the latest output & resume auto-follow
  Left/Right         browse input history (only at the line start/end)
  ESC                interrupt the running task""",
    },
    "zh": {
        "connecting": "連線 {url} …",
        "session_line": "session: {sid}   工作目錄: {wd}",
        "resumed": "已還原 session {sid}",
        "resume_tail": "── 對話已還原給模型;以下為最近幾則 ──",
        "config_restart_note": "提醒:參數變更需重啟或 /reset 後生效(只有 lang 即時)。",
        "welcome": "輸入訊息開始對話。/help 看指令,/exit 離開。",
        "prompt": "你 ▷ ",
        "agent_reply": "主代理 ◀ {out}",
        "bye": "再見。",
        "unknown_cmd": "未知指令:{cmd}(/help 看清單)",
        "tools_label": "主代理工具:{tools}",
        "kill_none": "目前沒有開啟中的持久終端。",
        "kill_list": "開啟中的終端 (pid  [代理]  備註):",
        "kill_usage": "用 /kill <pid> 強制關閉(只能關 falamus 啟動的)。",
        "compacted": "已壓縮上下文。",
        "no_compact": "未啟用自動壓縮。",
        "new_session": "已開新 session。",
        "no_sessions": "(無 session)",
        "saved_config": "已儲存設定。",
        "lang_set": "語言已切換:{lang}",
        "lang_usage": "用法:/lang <en|zh>",
        "resume_usage": "用法:/resume <sid>",
        "helper_loaded": "ok 已載入 falamus.md({n} 字)— 已注入主代理規範,首次對話即生效",
        "no_tools_chat": "! 純聊天模式 — 不送工具。資料夾沒有 falamus.md(不會新建)。",
        "no_tools_chat_rules": "! 純聊天模式 — 不送工具。發現 falamus.md({n} 字)— 已載入為規範(純聊天不會新建或修改)。",
        "ollama_pick_model": "ollama:目前沒有載入任何模型 — 請選擇要執行哪一個:",
        "ollama_pick_prompt": "  模型 [0-{n},預設 0]: ",
        "loading_model": "正在將模型 {model} 載入 ollama(首次載入可能需要一段時間)…",
        "confirm": "  !  {reason}\n  允許執行?[y/N]: ",
        "ask_workdir": "尚未設定工作目錄。請輸入一個路徑作為工作目錄(將在此新建/使用):\n> ",
        "workdir_missing": "工作目錄不存在:{wd}",
        "ask_create": "是否新建?[y/N]: ",
        "workdir_in_program": "工作目錄不可設在 helper 程式目錄內,請換一個。",
        "workdir_empty": "路徑不可為空。",
        "workdir_set": "已設定工作目錄:{wd}",
        "created_workdir": "已建立:{wd}",
        "cd_done": "已切換工作目錄:{wd}",
        "cd_bad": "不是目錄:{arg}",
        "bar_hint": "/help 看指令",
        "working": "處理中…",
        "streaming": "輸出中…",
        "standby": "待命中",
        "esc_interrupt": "esc 中斷",
        "busy_note": "(忙碌中 — 按 ESC 中斷)",
        "confirm_hint": "請在下方輸入 y / n",
        "interrupted": "[已中斷]",
        "ask_summary": "離開前要更新對話摘要到 falamus.md 嗎?",
        "saving_progress": "正在把對話摘要寫入 falamus.md …",
        "progress_saved": "進度已存入 falamus.md",
        "dev_on": "開發者模式 開啟 — 略過確認(破壞性指令仍需確認)",
        "dev_off": "開發者模式 關閉",
        "ctx_label": "ctx",
        "ctx_warn": "! 接近自動壓縮",
        "server_offline": "[server 離線] 連不上模型 server,請確認它在線。",
        "server_unreachable": (
            "連不上模型伺服器:{url}。\n"
            "  - 伺服器有啟動嗎?(llama.cpp:llama-server …   |   ollama:ollama serve)\n"
            "  - 設定位址:falamus --base-url http://HOST:PORT"
            "(或在 TUI 內:/config base_url http://HOST:PORT)"
        ),
        "server_down": "! SERVER 離線",
        "compacting": "壓縮上下文中…",
        "idle": "* 待命",
        "active_done": "作用中:{a} 完成:{d}",
        "help": """可用指令:
  /help              顯示本說明
  /version           顯示版本
  /b                 跳到最新輸出並恢復自動捲動(= Ctrl-G)
  /tools             列出主代理可用工具
  /kill [pid]        列出開啟中的終端 / 強制關閉指定 pid(--persistent-interactive-shell)
  /config            顯示目前設定(config.ini)
  /compact           手動壓縮上下文
  /reset             開一個新 session
  /sessions          列出本工作目錄的所有 session
  /resume <sid>      還原指定 session 接續對話
  /cd <path>         切換工作目錄(自動寫回 config.ini)
  /lang <en|zh>      切換語言
  /dev               切換開發者模式(略過確認)
  /save              儲存目前設定
  /exit              離開

操作鍵:
  Tab                切換焦點:輸出區 <-> 輸入區(切到「輸出區」才能捲動)
  Up/Down PgUp/PgDn  捲動輸出(先按 Tab);捲到底會恢復自動跟最新
  Ctrl-G             跳到最新輸出並恢復自動跟最新
  Left/Right         瀏覽輸入歷史(僅在行首/行尾)
  ESC                中斷進行中的任務""",
    },
}


def set_lang(lang: str) -> None:
    global _LANG
    _LANG = lang if lang in _MESSAGES else "en"


def get_lang() -> str:
    return _LANG


def t(key: str, **kw: object) -> str:
    s = _MESSAGES.get(_LANG, _MESSAGES["en"]).get(key) or _MESSAGES["en"].get(key, key)
    try:
        return s.format(**kw) if kw else s
    except (KeyError, IndexError):
        return s
