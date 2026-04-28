"""TmuxSyncCog — mirror externally-originated Claude turns into bound Discord threads.

Use case
--------
You're chatting with Claude Code in tmux on your laptop via ``claude --resume <id>``,
and you want the resulting research / answers to land in the matching Discord thread
so you can pick it up on your phone later.

By default ``ccdb`` only renders turns from subprocesses *it spawned itself* — turns
written to the session JSONL by an external ``claude --resume`` are invisible to
Discord. This Cog tails the JSONL and re-emits the missing turns into the bound
thread, with content-hash dedup to avoid double-posting Discord-originated turns.

Per-thread toggle
-----------------
Sync is **off** by default for every thread. Turn it on/off interactively::

    /tmux-sync on        — start mirroring this thread's session
    /tmux-sync off       — stop mirroring
    /tmux-sync status    — show current state + watch info

State is persisted in ``SettingsRepository`` under key ``tmux_sync:<thread_id>``.

Limitations
-----------
- Identical text sent within ``CONTENT_MATCH_WINDOW_S`` from both Discord and tmux
  will collapse into one (the tmux copy is treated as a Discord echo).
- Tool-use / thinking blocks render as plain text (no embeds / buttons).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import discord
from discord import app_commands
from discord.ext import commands

if TYPE_CHECKING:
    from ..bot import ClaudeDiscordBot
    from ..database.repository import SessionRepository
    from ..database.settings_repo import SettingsRepository

logger = logging.getLogger(__name__)


POLL_INTERVAL_S = 2.0
RESCAN_INTERVAL_S = 60.0
CONTENT_MATCH_WINDOW_S = 300.0
MAX_CHARS = 1800
PREFIX_USER = "\U0001f5a5️ **tmux →**"
PREFIX_ASSIST = "\U0001f5a5️ **Claude (tmux):**"

SETTING_KEY_PREFIX = "tmux_sync:"


def _setting_key(thread_id: int) -> str:
    return f"{SETTING_KEY_PREFIX}{thread_id}"


@dataclass
class _SessionState:
    """Per-session watch state. Only sessions whose thread has sync on are processed."""

    session_id: str
    thread_id: int
    jsonl_path: Path
    offset: int = 0
    seen_uuids: set[str] = field(default_factory=set)
    in_sync_cycle: bool | None = None
    enabled: bool = False  # mirrors the SettingsRepository value, refreshed each tick


def _hash_content(content: str) -> str:
    return hashlib.sha256(content.strip().encode("utf-8", errors="replace")).hexdigest()


def _extract_text(content: Any) -> str:
    """Pull plain text out of a `message.content` field (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                parts.append(f"[tool_use: {block.get('name', '?')}]")
            elif btype == "tool_result":
                raw = block.get("content", "")
                if isinstance(raw, list):
                    raw = _extract_text(raw)
                parts.append(f"[tool_result] {raw}")
            elif btype == "thinking":
                txt = block.get("thinking", "") or block.get("text", "")
                if txt:
                    parts.append(f"_(thinking)_ {txt[:400]}")
        return "\n".join(p for p in parts if p)
    return ""


def _chunk(text: str, limit: int = 1900) -> list[str]:
    if len(text) <= limit:
        return [text]
    out: list[str] = []
    buf = text
    while len(buf) > limit:
        cut = buf.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        out.append(buf[:cut])
        buf = buf[cut:].lstrip("\n")
    if buf:
        out.append(buf)
    return out


class TmuxSyncCog(commands.Cog):
    """Mirrors externally-originated JSONL turns into per-thread-toggled Discord threads."""

    def __init__(
        self,
        bot: ClaudeDiscordBot,
        *,
        session_repo: SessionRepository,
        settings_repo: SettingsRepository,
        cli_sessions_path: Path,
    ) -> None:
        self.bot = bot
        self.session_repo = session_repo
        self.settings_repo = settings_repo
        self.cli_sessions_path = cli_sessions_path
        self._watched: dict[str, _SessionState] = {}
        self._recent_discord: dict[int, deque[tuple[float, str]]] = {}
        self._poll_task: asyncio.Task[None] | None = None
        self._rescan_task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def cog_load(self) -> None:
        logger.info(
            "TmuxSyncCog starting (poll=%.1fs, dir=%s)",
            POLL_INTERVAL_S,
            self.cli_sessions_path,
        )
        await self._initial_scan()
        self._poll_task = asyncio.create_task(self._poll_loop(), name="tmux-sync-poll")
        self._rescan_task = asyncio.create_task(self._rescan_loop(), name="tmux-sync-rescan")

    async def cog_unload(self) -> None:
        self._stopping.set()
        for task in (self._poll_task, self._rescan_task):
            if task is not None:
                task.cancel()
        self._poll_task = None
        self._rescan_task = None

    # ------------------------------------------------------------------
    # Discovery & baseline
    # ------------------------------------------------------------------

    async def _initial_scan(self) -> None:
        records = await self.session_repo.list_all(limit=200)
        # list_all returns DESC by last_used_at, so the first time we see a
        # session_id corresponds to its most-recently-used thread. Skip later
        # duplicates (e.g. /resume creates a 2nd thread on the same session).
        seen_sessions: set[str] = set()
        for rec in records:
            if rec.session_id in seen_sessions:
                continue
            seen_sessions.add(rec.session_id)
            await self._watch(rec.session_id, rec.thread_id)
        logger.info("TmuxSyncCog tracking %d sessions (sync state per-thread)", len(self._watched))

    def _find_jsonl(self, session_id: str) -> Path | None:
        hits = list(self.cli_sessions_path.glob(f"**/{session_id}.jsonl"))
        return hits[0] if hits else None

    async def _watch(self, session_id: str, thread_id: int) -> _SessionState | None:
        """Add a session's watch state with baseline=current-EOF.

        Passive — if the session is already watched, returns the existing
        state untouched. Callers that need to switch which thread receives
        mirrored output for a session should use ``_rebind_thread()``.

        Always baselines the JSONL (records existing uuids, offset = file size)
        so that history is never re-posted, even when sync is later flipped on.
        """
        existing = self._watched.get(session_id)
        if existing is not None:
            return existing
        path = self._find_jsonl(session_id)
        if path is None:
            return None

        state = _SessionState(session_id=session_id, thread_id=thread_id, jsonl_path=path)
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    uid = obj.get("uuid")
                    if uid:
                        state.seen_uuids.add(uid)
            state.offset = path.stat().st_size
        except OSError:
            logger.debug("Baseline read failed for %s", path, exc_info=True)

        state.enabled = await self._is_enabled(thread_id)
        self._watched[session_id] = state
        return state

    async def _rebind_thread(self, session_id: str, thread_id: int) -> _SessionState | None:
        """Explicitly point a watched session at a different thread.

        Called from the slash command path when a user runs ``/tmux-sync on``
        in a thread that already shares its session_id with another thread
        (e.g. the same session was opened in multiple threads via ``/resume``).
        The user's most recent action wins — future mirrored output goes to
        the thread they're currently in.

        If the session has not yet been watched, falls through to ``_watch``
        which creates the initial state with the requested thread_id.
        """
        state = self._watched.get(session_id)
        if state is None:
            return await self._watch(session_id, thread_id)
        if state.thread_id != thread_id:
            logger.info(
                "TmuxSyncCog rebinding session %s from thread %d to %d",
                session_id[:8],
                state.thread_id,
                thread_id,
            )
            state.thread_id = thread_id
        return state

    async def _is_enabled(self, thread_id: int) -> bool:
        val = await self.settings_repo.get(_setting_key(thread_id))
        return val == "on"

    # ------------------------------------------------------------------
    # Discord listeners
    # ------------------------------------------------------------------

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return
        thread_id = message.channel.id
        rec = await self.session_repo.get(thread_id)
        if rec is None:
            return
        # Always record content for dedup, even when sync is off — so when the
        # user flips sync on mid-conversation, we already know the recent
        # Discord-originated content.
        self._record_discord_msg(thread_id, message.content or "")
        if rec.session_id not in self._watched:
            await self._watch(rec.session_id, thread_id)

    def _record_discord_msg(self, thread_id: int, content: str) -> None:
        if not content:
            return
        h = _hash_content(content)
        dq = self._recent_discord.setdefault(thread_id, deque(maxlen=50))
        dq.append((time.time(), h))

    def _is_discord_originated(self, thread_id: int, content: str) -> bool:
        dq = self._recent_discord.get(thread_id)
        if not dq:
            return False
        h = _hash_content(content)
        now = time.time()
        while dq and now - dq[0][0] > CONTENT_MATCH_WINDOW_S:
            dq.popleft()
        return any(hh == h for _, hh in dq)

    # ------------------------------------------------------------------
    # Slash commands  /tmux-sync <action>
    # ------------------------------------------------------------------

    tmux_sync_group = app_commands.Group(
        name="tmux-sync",
        description="Mirror tmux-originated Claude turns into this thread.",
    )

    @tmux_sync_group.command(name="on", description="Enable tmux mirroring for the current thread.")
    async def cmd_on(self, interaction: discord.Interaction) -> None:
        thread_id = interaction.channel_id
        if thread_id is None:
            await interaction.response.send_message("Cannot resolve channel.", ephemeral=True)
            return
        rec = await self.session_repo.get(thread_id)
        if rec is None:
            await interaction.response.send_message(
                "This thread is not bound to a Claude session yet. Send a message first.",
                ephemeral=True,
            )
            return

        await self.settings_repo.set(_setting_key(thread_id), "on")
        # Bind/rebind the watch state to THIS thread so future mirrors route
        # correctly even if another thread previously claimed this session_id.
        state = await self._rebind_thread(rec.session_id, thread_id)
        if state is not None:
            state.enabled = True
            # Re-baseline so we don't dump pre-existing history.
            with contextlib.suppress(OSError):
                state.offset = state.jsonl_path.stat().st_size
            state.in_sync_cycle = None  # will resolve on next user turn

        await interaction.response.send_message(
            f"\U0001f5a5️ Tmux mirroring **enabled** for this thread.\n"
            f"Session `{rec.session_id[:8]}…` will mirror new external turns. "
            f"Use `/tmux-sync off` to stop.",
            ephemeral=True,
        )

    @tmux_sync_group.command(
        name="off", description="Disable tmux mirroring for the current thread."
    )
    async def cmd_off(self, interaction: discord.Interaction) -> None:
        thread_id = interaction.channel_id
        if thread_id is None:
            await interaction.response.send_message("Cannot resolve channel.", ephemeral=True)
            return
        await self.settings_repo.delete(_setting_key(thread_id))
        rec = await self.session_repo.get(thread_id)
        if rec is not None:
            state = self._watched.get(rec.session_id)
            if state is not None:
                state.enabled = False
        await interaction.response.send_message(
            "Tmux mirroring **disabled** for this thread.", ephemeral=True
        )

    @tmux_sync_group.command(
        name="status", description="Show tmux mirroring state for this thread."
    )
    async def cmd_status(self, interaction: discord.Interaction) -> None:
        thread_id = interaction.channel_id
        if thread_id is None:
            await interaction.response.send_message("Cannot resolve channel.", ephemeral=True)
            return
        rec = await self.session_repo.get(thread_id)
        enabled = await self._is_enabled(thread_id)

        state_label = "on ✅" if enabled else "off"
        lines = [f"**Tmux sync:** {state_label}"]
        if rec is None:
            lines.append("Thread is not bound to a Claude session yet.")
        else:
            lines.append(f"**Session:** `{rec.session_id[:8]}…`")
            state = self._watched.get(rec.session_id)
            if state is None:
                lines.append("_Not yet watched (will be picked up shortly)._")
            else:
                lines.append(f"**JSONL:** `{state.jsonl_path.name}`")
                lines.append(f"**Offset:** {state.offset} bytes")
                lines.append(f"**Seen UUIDs:** {len(state.seen_uuids)}")
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    @tmux_sync_group.command(
        name="list", description="List all threads with tmux mirroring enabled."
    )
    async def cmd_list(self, interaction: discord.Interaction) -> None:
        all_settings = await self.settings_repo.get_all()
        on_threads = [
            int(k[len(SETTING_KEY_PREFIX) :])
            for k, v in all_settings.items()
            if k.startswith(SETTING_KEY_PREFIX) and v == "on"
        ]
        if not on_threads:
            await interaction.response.send_message(
                "No threads have tmux sync enabled.", ephemeral=True
            )
            return
        lines = [f"**{len(on_threads)} thread(s) with tmux sync on:**"]
        for tid in on_threads[:25]:
            rec = await self.session_repo.get(tid)
            sid = f"`{rec.session_id[:8]}…`" if rec else "_(unbound)_"
            lines.append(f"• <#{tid}> — {sid}")
        if len(on_threads) > 25:
            lines.append(f"_… and {len(on_threads) - 25} more_")
        await interaction.response.send_message("\n".join(lines), ephemeral=True)

    # ------------------------------------------------------------------
    # Background poll
    # ------------------------------------------------------------------

    async def _rescan_loop(self) -> None:
        try:
            while not self._stopping.is_set():
                await asyncio.sleep(RESCAN_INTERVAL_S)
                try:
                    records = await self.session_repo.list_all(limit=200)
                    for rec in records:
                        if rec.session_id not in self._watched:
                            await self._watch(rec.session_id, rec.thread_id)
                    # Refresh enabled flags from DB (in case of cross-bot edits)
                    for state in self._watched.values():
                        state.enabled = await self._is_enabled(state.thread_id)
                except Exception:
                    logger.exception("TmuxSyncCog rescan failed")
        except asyncio.CancelledError:
            pass

    async def _poll_loop(self) -> None:
        try:
            while not self._stopping.is_set():
                await asyncio.sleep(POLL_INTERVAL_S)
                for state in list(self._watched.values()):
                    if not state.enabled:
                        # Advance the offset so when sync is later turned on we
                        # don't replay accumulated tmux activity.
                        with contextlib.suppress(OSError):
                            state.offset = state.jsonl_path.stat().st_size
                        continue
                    try:
                        await self._drain_session(state)
                    except Exception:
                        logger.exception("TmuxSyncCog drain failed for %s", state.session_id[:8])
        except asyncio.CancelledError:
            pass

    async def _drain_session(self, state: _SessionState) -> None:
        try:
            size = state.jsonl_path.stat().st_size
        except FileNotFoundError:
            return
        if size <= state.offset:
            return

        with state.jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(state.offset)
            chunk_bytes = f.read()
            state.offset = f.tell()

        for line in chunk_bytes.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            await self._handle_turn(state, obj)

    async def _handle_turn(self, state: _SessionState, obj: dict) -> None:
        ttype = obj.get("type")
        if ttype not in ("user", "assistant"):
            return
        if obj.get("isMeta"):
            return
        uid = obj.get("uuid")
        if uid and uid in state.seen_uuids:
            return
        if uid:
            state.seen_uuids.add(uid)

        text = _extract_text((obj.get("message") or {}).get("content", ""))
        if not text.strip():
            return
        if text.lstrip().startswith("<"):
            return

        if ttype == "user":
            if self._is_discord_originated(state.thread_id, text):
                state.in_sync_cycle = False
                return
            state.in_sync_cycle = True
            await self._post(state.thread_id, role="user", text=text)
            return

        # assistant — only mirror when we're inside a tmux-originated cycle
        if state.in_sync_cycle is True:
            await self._post(state.thread_id, role="assistant", text=text)

    async def _post(self, thread_id: int, *, role: str, text: str) -> None:
        channel = self.bot.get_channel(thread_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(thread_id)
            except (discord.NotFound, discord.Forbidden):
                logger.warning("TmuxSyncCog cannot reach channel %d", thread_id)
                return

        prefix = PREFIX_USER if role == "user" else PREFIX_ASSIST
        body = text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n… _(truncated)_"
        first = True
        for piece in _chunk(f"{prefix}\n{body}" if first else body):
            try:
                await channel.send(piece)
            except discord.HTTPException:
                logger.exception("TmuxSyncCog send failed to %d", thread_id)
                return
            first = False
