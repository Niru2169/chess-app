from __future__ import annotations

import subprocess
import threading
import queue
import time
from dataclasses import dataclass
from typing import Iterable

@dataclass(frozen=True)
class UciBestMove:
    move: str
    ponder: str | None = None

@dataclass(frozen=True)
class DisplayInfo:
    fen: str | None
    key: str | None
    checkers_raw: str | None

class UciError(RuntimeError):
    """UCI communication error or engine problem."""

# Bringing an engine up is the slow part: a cold Stockfish has to map its binary
# and initialise a large NNUE network, which easily takes a couple of seconds on a
# busy machine (and the very first start of a process is the slowest). Budget the
# lifecycle commands generously so a slow start can never forfeit a match; the
# per-move budgets below stay tight.
_STARTUP_TIMEOUT_S = 10.0

class UciProcess:
    """UCI engine wrapper: a single subprocess plus a reader thread.

    Important:
        This component is intentionally minimal (stdlib only).
        It works well alongside a single Django worker for the MVP.
    """

    def __init__(self, exe_path: str) -> None:
        self._exe_path = exe_path
        self._proc: subprocess.Popen[str] | None = None
        self._lines: "queue.Queue[str]" = queue.Queue()
        self._reader: threading.Thread | None = None
        self._write_lock = threading.Lock()

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(
            [self._exe_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        self._uci_handshake()

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            with self._write_lock:
                if proc.stdin:
                    proc.stdin.write("quit\n")
                    proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass

    def set_option(self, name: str, value: str | int | bool) -> None:
        self._ensure()
        v = "true" if value is True else "false" if value is False else str(value)
        self._send(f"setoption name {name} value {v}")
        self.is_ready(timeout_s=_STARTUP_TIMEOUT_S)

    def new_game(self) -> None:
        self._ensure()
        self._send("ucinewgame")
        self.is_ready(timeout_s=_STARTUP_TIMEOUT_S)

    def position_fen(self, fen: str, moves: Iterable[str] | None = None) -> None:
        self._ensure()
        cmd = f"position fen {fen}"
        if moves:
            cmd += " moves " + " ".join(moves)
        self._send(cmd)

    def go_movetime(self, movetime_ms: int, timeout_s: float = 5.0) -> UciBestMove:
        self._ensure()
        self._send(f"go movetime {movetime_ms}")
        line = self._wait_for_prefix("bestmove", timeout_s=timeout_s)
        parts = line.split()
        if len(parts) < 2:
            raise UciError(f"Invalid bestmove line: {line!r}")
        move = parts[1]
        ponder = None
        if len(parts) >= 4 and parts[2] == "ponder":
            ponder = parts[3]
        return UciBestMove(move=move, ponder=ponder)

    def display(self, timeout_s: float = 2.0) -> DisplayInfo:
        """Run Stockfish's 'd' command and extract the Fen/Key/Checkers information.

        Note:
            'd' is not part of standard UCI, but it is widely available in Stockfish.
        """
        self._ensure()
        self._send("d")
        # Sync with isready so we know output is complete.
        self._send("isready")
        fen = None
        key = None
        checkers = None
        start = time.monotonic()
        while True:
            remaining = timeout_s - (time.monotonic() - start)
            if remaining <= 0:
                raise UciError("Timeout while reading the 'd' output.")
            try:
                line = self._lines.get(timeout=remaining).strip()
            except queue.Empty:
                raise UciError("Timeout while reading the 'd' output.")
            if line.startswith("Fen:"):
                fen = line.replace("Fen:", "", 1).strip()
            elif line.startswith("Key:"):
                key = line.replace("Key:", "", 1).strip()
            elif line.startswith("Checkers:"):
                checkers = line.replace("Checkers:", "", 1).strip()
            elif line == "readyok":
                break
        return DisplayInfo(fen=fen, key=key, checkers_raw=checkers)

    def is_ready(self, timeout_s: float = _STARTUP_TIMEOUT_S) -> None:
        self._ensure()
        self._send("isready")
        _ = self._wait_for_exact("readyok", timeout_s=timeout_s)

    # ----- internal helpers -----

    def _ensure(self) -> None:
        if self._proc is None:
            raise UciError("The UCI process has not been started. Call start().")

    def _send(self, cmd: str) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise UciError("Engine stdin is not available.")
        with self._write_lock:
            proc.stdin.write(cmd.strip() + "\n")
            proc.stdin.flush()

    def _uci_handshake(self) -> None:
        self._send("uci")
        _ = self._wait_for_exact("uciok", timeout_s=_STARTUP_TIMEOUT_S)

    def _read_line(self, remaining_s: float, context: str) -> str:
        """Return the next engine line, or an empty string if none arrived yet.

        Raises:
            UciError: If the engine process has died, which is worth reporting
                immediately: an engine that cannot allocate its network should not
                look like a slow engine until the whole budget runs out.
        """
        try:
            return self._lines.get(timeout=min(remaining_s, 0.25)).strip()
        except queue.Empty:
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise UciError(
                    f"Engine exited (code {proc.returncode}) while waiting for {context}."
                )
            return ""

    def _wait_for_exact(self, token: str, timeout_s: float) -> str:
        start = time.monotonic()
        while True:
            remaining = timeout_s - (time.monotonic() - start)
            if remaining <= 0:
                raise UciError(f"Timeout while waiting for {token!r}.")
            if self._read_line(remaining, repr(token)) == token:
                return token

    def _wait_for_prefix(self, prefix: str, timeout_s: float) -> str:
        start = time.monotonic()
        while True:
            remaining = timeout_s - (time.monotonic() - start)
            if remaining <= 0:
                raise UciError(f"Timeout while waiting for a {prefix!r} line.")
            line = self._read_line(remaining, f"a {prefix!r} line")
            if line.startswith(prefix):
                return line

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            self._lines.put(raw.rstrip("\n"))
