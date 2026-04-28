"""Tests for TmuxSyncCog — pure logic + state machine behavior."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from claude_discord.cogs.tmux_sync import (
    CONTENT_MATCH_WINDOW_S,
    SETTING_KEY_PREFIX,
    TmuxSyncCog,
    _chunk,
    _extract_text,
    _hash_content,
    _SessionState,
    _setting_key,
)

# ---------------------------------------------------------------------------
# Pure logic
# ---------------------------------------------------------------------------


class TestExtractText:
    def test_string_content(self):
        assert _extract_text("hello") == "hello"

    def test_text_blocks(self):
        blocks = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert _extract_text(blocks) == "a\nb"

    def test_tool_use_block_renders_marker(self):
        blocks = [{"type": "tool_use", "name": "Bash"}]
        assert _extract_text(blocks) == "[tool_use: Bash]"

    def test_tool_result_block_with_string(self):
        blocks = [{"type": "tool_result", "content": "stdout: ok"}]
        assert _extract_text(blocks) == "[tool_result] stdout: ok"

    def test_tool_result_block_with_nested_blocks(self):
        blocks = [{"type": "tool_result", "content": [{"type": "text", "text": "nested"}]}]
        assert "nested" in _extract_text(blocks)

    def test_thinking_block_truncated(self):
        long_text = "x" * 1000
        blocks = [{"type": "thinking", "thinking": long_text}]
        out = _extract_text(blocks)
        assert "(thinking)" in out
        # truncation at 400 chars + prefix
        assert len(out) < 500

    def test_empty_input(self):
        assert _extract_text("") == ""
        assert _extract_text([]) == ""
        assert _extract_text({"unexpected": "object"}) == ""

    def test_skips_unknown_blocks(self):
        blocks = [{"type": "image", "data": "..."}, {"type": "text", "text": "kept"}]
        assert _extract_text(blocks) == "kept"


class TestChunk:
    def test_short_text_unchanged(self):
        assert _chunk("hello") == ["hello"]

    def test_breaks_on_newline_when_possible(self):
        text = "a" * 1000 + "\n" + "b" * 1000
        chunks = _chunk(text, limit=1500)
        assert len(chunks) == 2
        assert chunks[0].endswith("a")
        assert chunks[1].startswith("b")

    def test_hard_cut_when_no_break_near_end(self):
        text = "x" * 5000
        chunks = _chunk(text, limit=1900)
        assert all(len(c) <= 1900 for c in chunks)
        assert sum(len(c) for c in chunks) == len(text)


class TestHashContent:
    def test_strip_normalization(self):
        assert _hash_content("hello") == _hash_content("  hello  ")
        assert _hash_content("hello") == _hash_content("hello\n")

    def test_different_content_different_hash(self):
        assert _hash_content("a") != _hash_content("b")


def test_setting_key_format():
    assert _setting_key(12345) == f"{SETTING_KEY_PREFIX}12345"


# ---------------------------------------------------------------------------
# Cog state machine — drains an actual JSONL fixture
# ---------------------------------------------------------------------------


def _write_jsonl_lines(path: Path, lines: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


def _make_cog(tmp_path: Path) -> tuple[TmuxSyncCog, MagicMock, MagicMock]:
    bot = MagicMock()
    bot.get_channel = MagicMock(return_value=MagicMock(send=AsyncMock()))
    session_repo = MagicMock()
    session_repo.list_all = AsyncMock(return_value=[])
    session_repo.get = AsyncMock(return_value=None)
    settings_repo = MagicMock()
    settings_repo.get = AsyncMock(return_value=None)
    settings_repo.set = AsyncMock()
    settings_repo.delete = AsyncMock()
    settings_repo.get_all = AsyncMock(return_value={})
    cog = TmuxSyncCog(
        bot,
        session_repo=session_repo,
        settings_repo=settings_repo,
        cli_sessions_path=tmp_path,
    )
    return cog, session_repo, settings_repo


def _user_turn(uid: str, content: str) -> dict:
    return {"type": "user", "uuid": uid, "message": {"content": content}}


def _assist_turn(uid: str, content: str) -> dict:
    return {"type": "assistant", "uuid": uid, "message": {"content": content}}


@pytest.mark.asyncio
async def test_baseline_skips_history(tmp_path: Path):
    """Sessions watched after they have history should NOT replay it."""
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    _write_jsonl_lines(jsonl, [_user_turn("u1", "old"), _assist_turn("a1", "old reply")])

    cog, _, _ = _make_cog(tmp_path)
    state = await cog._watch(sid, thread_id=999)
    assert state is not None
    # Baseline captured both uuids
    assert "u1" in state.seen_uuids
    assert "a1" in state.seen_uuids
    # Offset moved to EOF — drain finds nothing new
    state.enabled = True
    await cog._drain_session(state)
    cog.bot.get_channel.return_value.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_external_user_turn_is_mirrored(tmp_path: Path):
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, _, _ = _make_cog(tmp_path)
    state = await cog._watch(sid, thread_id=999)
    assert state is not None
    state.enabled = True

    # Append a tmux-originated turn pair
    _write_jsonl_lines(
        jsonl,
        [_user_turn("u2", "from tmux"), _assist_turn("a2", "claude reply")],
    )
    await cog._drain_session(state)

    sent = cog.bot.get_channel.return_value.send
    assert sent.await_count == 2
    contents = [call.args[0] for call in sent.await_args_list]
    assert any("from tmux" in c for c in contents)
    assert any("claude reply" in c for c in contents)


@pytest.mark.asyncio
async def test_discord_originated_turn_is_swallowed(tmp_path: Path):
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, _, _ = _make_cog(tmp_path)
    state = await cog._watch(sid, thread_id=999)
    assert state is not None
    state.enabled = True

    # Pretend Discord just received this exact text
    cog._record_discord_msg(thread_id=999, content="from discord")

    _write_jsonl_lines(
        jsonl,
        [
            _user_turn("u3", "from discord"),
            _assist_turn("a3", "this assistant should NOT be mirrored"),
        ],
    )
    await cog._drain_session(state)

    sent = cog.bot.get_channel.return_value.send
    sent.assert_not_awaited()
    assert state.in_sync_cycle is False


@pytest.mark.asyncio
async def test_state_machine_resets_on_next_user_turn(tmp_path: Path):
    """After a Discord cycle, a tmux cycle should resume mirroring."""
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, _, _ = _make_cog(tmp_path)
    state = await cog._watch(sid, thread_id=999)
    assert state is not None
    state.enabled = True

    cog._record_discord_msg(thread_id=999, content="discord-1")

    _write_jsonl_lines(
        jsonl,
        [
            _user_turn("u4", "discord-1"),  # SKIP cycle
            _assist_turn("a4", "should be skipped"),
            _user_turn("u5", "tmux-2"),  # SYNC cycle
            _assist_turn("a5", "should be mirrored"),
        ],
    )
    await cog._drain_session(state)

    sent = cog.bot.get_channel.return_value.send
    contents = [call.args[0] for call in sent.await_args_list]
    assert any("tmux-2" in c for c in contents)
    assert any("should be mirrored" in c for c in contents)
    assert not any("should be skipped" in c for c in contents)


@pytest.mark.asyncio
async def test_disabled_thread_not_mirrored_but_offset_advances(tmp_path: Path):
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, _, _ = _make_cog(tmp_path)
    state = await cog._watch(sid, thread_id=999)
    assert state is not None
    state.enabled = False  # disabled

    _write_jsonl_lines(jsonl, [_user_turn("u6", "tmux while off")])

    # Disabled-state branch in _poll_loop just advances offset; emulate that
    state.offset = state.jsonl_path.stat().st_size

    # Now turn on — there should be NO replay because offset already past EOF
    state.enabled = True
    await cog._drain_session(state)
    cog.bot.get_channel.return_value.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_watch_is_passive_first_call_wins(tmp_path: Path):
    """_watch is the passive baseline path. First call wins; subsequent calls
    return the existing state untouched. Active rebinds use _rebind_thread.
    """
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, _, _ = _make_cog(tmp_path)
    state_a = await cog._watch(sid, thread_id=111)
    assert state_a is not None
    assert state_a.thread_id == 111

    # Second _watch must NOT overwrite (e.g. initial_scan iterating older rows
    # of the same session_id should not clobber the more-recent thread).
    state_b = await cog._watch(sid, thread_id=222)
    assert state_b is state_a
    assert state_b.thread_id == 111  # untouched


@pytest.mark.asyncio
async def test_rebind_thread_changes_active_thread(tmp_path: Path):
    """_rebind_thread is the explicit path used by /tmux-sync on."""
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, _, _ = _make_cog(tmp_path)
    state = await cog._watch(sid, thread_id=111)
    assert state is not None and state.thread_id == 111

    rebound = await cog._rebind_thread(sid, thread_id=222)
    assert rebound is state
    assert rebound.thread_id == 222


@pytest.mark.asyncio
async def test_rebind_thread_creates_state_when_unwatched(tmp_path: Path):
    """If the session is brand new, _rebind_thread should fall through to _watch."""
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, _, _ = _make_cog(tmp_path)
    state = await cog._rebind_thread(sid, thread_id=111)
    assert state is not None
    assert state.thread_id == 111
    assert sid in cog._watched


@pytest.mark.asyncio
async def test_initial_scan_dedupes_by_session_id_keeping_most_recent(tmp_path: Path):
    """When SessionRepository contains two rows with the same session_id (e.g.
    /resume created a fresh thread on an existing session), initial_scan must
    bind the watch to the most-recently-used thread. list_all returns DESC by
    last_used_at, so the FIRST row wins.
    """
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, session_repo, _ = _make_cog(tmp_path)
    # Newest first (DESC by last_used_at): NEW thread row, then OLD thread row
    session_repo.list_all = AsyncMock(
        return_value=[
            MagicMock(session_id=sid, thread_id=222),  # most recent
            MagicMock(session_id=sid, thread_id=111),  # older
        ]
    )
    await cog._initial_scan()
    state = cog._watched[sid]
    assert state.thread_id == 222  # most-recent thread, not 111


@pytest.mark.asyncio
async def test_cmd_on_rebinds_thread_id_for_existing_state(tmp_path: Path):
    """If the cog already watched this session under an older thread (e.g. via
    initial scan), running /tmux-sync on from a newer thread that shares the
    same session_id must redirect future mirrors to the newer thread.
    """
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, session_repo, _ = _make_cog(tmp_path)

    # Pre-seed state with the OLD thread (as initial_scan would do)
    await cog._watch(sid, thread_id=111)

    # User runs /tmux-sync on from the NEW thread
    session_repo.get = AsyncMock(return_value=MagicMock(session_id=sid, thread_id=222))
    interaction = MagicMock()
    interaction.channel_id = 222
    interaction.response.send_message = AsyncMock()
    await cog.cmd_on.callback(cog, interaction)

    state = cog._watched[sid]
    assert state.thread_id == 222
    assert state.enabled is True


@pytest.mark.asyncio
async def test_mirror_routes_to_rebound_thread_after_resume(tmp_path: Path):
    """End-to-end: after rebinding, a tmux turn must hit the new thread, not the old one."""
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    jsonl.touch()

    cog, session_repo, _ = _make_cog(tmp_path)

    # Old thread first
    await cog._watch(sid, thread_id=111)

    # Rebind via cmd_on from new thread
    session_repo.get = AsyncMock(return_value=MagicMock(session_id=sid, thread_id=222))
    interaction = MagicMock()
    interaction.channel_id = 222
    interaction.response.send_message = AsyncMock()
    await cog.cmd_on.callback(cog, interaction)

    # Append a tmux turn
    _write_jsonl_lines(
        jsonl,
        [_user_turn("u_new", "from tmux"), _assist_turn("a_new", "claude reply")],
    )
    await cog._drain_session(cog._watched[sid])

    # bot.get_channel must have been called with the NEW thread id (222)
    called_ids = [c.args[0] for c in cog.bot.get_channel.call_args_list]
    assert 222 in called_ids
    assert 111 not in called_ids


@pytest.mark.asyncio
async def test_dedup_window_evicts_stale_entries(tmp_path: Path):
    cog, _, _ = _make_cog(tmp_path)
    cog._record_discord_msg(thread_id=1, content="hello")
    # Force the timestamp to be older than the window
    dq = cog._recent_discord[1]
    stale_ts = time.time() - (CONTENT_MATCH_WINDOW_S + 10)
    dq[0] = (stale_ts, dq[0][1])
    assert cog._is_discord_originated(1, "hello") is False


# ---------------------------------------------------------------------------
# Slash command — happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cmd_on_persists_and_baselines(tmp_path: Path):
    sid = "abc-123"
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    jsonl = proj_dir / f"{sid}.jsonl"
    _write_jsonl_lines(jsonl, [_user_turn("u_old", "before turning on")])

    cog, session_repo, settings_repo = _make_cog(tmp_path)
    session_repo.get = AsyncMock(return_value=MagicMock(session_id=sid, thread_id=999))

    interaction = MagicMock()
    interaction.channel_id = 999
    interaction.response.send_message = AsyncMock()

    await cog.cmd_on.callback(cog, interaction)

    settings_repo.set.assert_awaited_once_with(_setting_key(999), "on")
    interaction.response.send_message.assert_awaited_once()
    # The session should be watched and its offset at EOF
    state = cog._watched[sid]
    assert state.enabled is True
    assert state.offset == jsonl.stat().st_size


@pytest.mark.asyncio
async def test_cmd_off_clears_setting(tmp_path: Path):
    cog, session_repo, settings_repo = _make_cog(tmp_path)
    session_repo.get = AsyncMock(return_value=None)

    interaction = MagicMock()
    interaction.channel_id = 999
    interaction.response.send_message = AsyncMock()

    await cog.cmd_off.callback(cog, interaction)

    settings_repo.delete.assert_awaited_once_with(_setting_key(999))


@pytest.mark.asyncio
async def test_cmd_status_reports_state(tmp_path: Path):
    cog, session_repo, settings_repo = _make_cog(tmp_path)
    session_repo.get = AsyncMock(return_value=None)
    settings_repo.get = AsyncMock(return_value="on")

    interaction = MagicMock()
    interaction.channel_id = 999
    interaction.response.send_message = AsyncMock()

    await cog.cmd_status.callback(cog, interaction)

    sent = interaction.response.send_message.await_args.args[0]
    assert "on" in sent
    assert "not bound" in sent.lower()


def test_session_state_dataclass_defaults():
    state = _SessionState(session_id="x", thread_id=1, jsonl_path=Path("/tmp/x"))
    assert state.offset == 0
    assert state.in_sync_cycle is None
    assert state.enabled is False
    assert state.seen_uuids == set()
