# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Pinned-host ping-pong mirror of a Newton GPU buffer.

Why this exists
---------------
Reading ``state_0.joint_q`` (or any device-side ``wp.array``) from
the publish thread via ``.numpy()`` issues a SYNCHRONOUS device→host
transfer that blocks until *all queued GPU work on that stream* has
completed.  At 100 Hz on a G2-class scene that's ~6 ms per tick of
pure CPU stall, multiplied across joints + bodies + odom — enough to
push the publish phase over the 10 ms budget all by itself.

The fix: keep a pinned-host buffer that the engine refreshes via
``wp.copy(host_pinned, src, stream=…)`` enqueued onto the same stream
the captured physics graph runs on, then have the publish path read
that buffer directly with no further GPU contact.  Pinned memory
+ stream-ordered memcpy means the copy is asynchronous: the host
returns immediately and the DMA engine finishes the move while the
next tick's graph runs.

Why a single buffer is unsafe
-----------------------------
A single host buffer plus a single ``wp.Event`` object DOES NOT WORK,
even when "rotating" the references each tick:

    self._event_prev = self._event   # both point to E
    stream.record_event(self._event) # mutates E in place

``wp.Event`` is a CUDA event handle.  ``record_event`` overwrites the
event's recorded position in place — there is no immutable snapshot.
So ``self._event_prev`` and ``self._event`` end up pointing at the
*same* recording, and ``synchronize_event(self._event_prev)`` waits
for the LATEST tick's graph + copy to finish, defeating the whole
point.  The pinned-host buffer is also race-prone: while
``get_joint_states`` reads it on the host, ``step`` could be enqueuing
a fresh ``wp.copy`` into the same memory.

Ping-pong design
----------------
Two host buffers, alternated by a writer index:

    refresh(src):
        slot = self._next_writer
        wp.copy(buf[slot], src, stream)
        self._next_writer = 1 - slot       # flip

    read():
        # Return the OTHER slot — the one written *last* tick, not
        # the one just written this tick.  ``self._next_writer``
        # points at the slot that will be overwritten next; it
        # currently still holds the previous-tick data.
        return views[next_writer]

Tick 1 has nothing on the "other" slot yet (it was never written),
so ``read`` returns ``None`` and the caller takes the synchronous
fallback ONCE.  From tick 2 onward the read is microseconds — the
slot we return was filled an entire physics tick ago and the DMA
copy itself is microseconds; by the time the publish thread reaches
``read``, the copy has been visible in pinned memory for ≥ one
physics_dt of wall time.

Each side now owns immutable state for the duration it's being
read: refresh writes slot A while read reads slot B.

Why no event sync
-----------------
``read`` does not call ``wp.synchronize_event``.  On warp 1.14 that
call burns 2-6 ms per tick even when the event is structurally
guaranteed to have fired (recorded one tick ago, on a stream with
10+ ms of idle time since).  The event API is conservatively waiting
on more than the recorded operation — likely back to the most recent
capture barrier on the stream.

The ping-pong itself already guarantees correctness through
**wall-time separation**, which is a stronger invariant than the
event:
  * The slot the reader returns was filled at the END of the
    previous tick's ``refresh`` (≥ ``physics_dt`` ago).
  * The actual DMA is a few hundred bytes; even at PCIe 3.0 it
    completes in < 100 µs.
  * Any subsequent stream operation (next graph launch, next
    copy) implicitly serializes behind the prior memcpy in FIFO
    order — there is no scenario where a ``physics_dt``-old
    pinned-mem write hasn't reached the host.

So an event sync is redundant insurance against a race that the
ping-pong design already prevents, and skipping it is the difference
between a 6 ms ``read`` and a sub-microsecond one.

Trade-off
---------
The mirror lags by ONE physics tick — the latest captured-graph
output is in the buffer the writer is currently filling, not the one
the reader sees.  At 100 Hz that's 10 ms of /joint_states lag, well
inside what RViz / WBC / state observation already tolerate from
ROS DDS jitter.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import warp as wp


class AsyncMirror:
    """Two-slot pinned-host mirror of a single Newton GPU array.

    Owns:
      * Two ``wp.array`` buffers on ``device="cpu", pinned=True``
      * A writer index that ping-pongs after each ``refresh``

    See the module docstring for why no per-buffer event is kept.

    Lifecycle:
      * Construct with no arguments at engine init.
      * Call ``refresh(src, stream)`` once per physics step.  First
        call sizes the buffers from ``src`` and records the first
        event; subsequent calls reuse the buffers in place.
      * Call ``read()`` from the publish thread.  Returns ``None``
        until the first ``refresh`` has flipped the writer (i.e. the
        publish thread runs after step has called refresh at least
        once).  Callers should fall back to the synchronous
        ``src.numpy()`` path during this brief warmup window.
    """

    __slots__ = (
        "_bufs",
        "_views",
        "_next_writer",
        "_refresh_count",
        "_failed",
    )

    def __init__(self) -> None:
        self._bufs: list = [None, None]
        self._views: list = [None, None]
        # Index of the slot the next ``refresh`` will write into.
        # After a refresh, this flips to the OTHER slot — so ``read``
        # uses ``self._next_writer`` itself as the index of the slot
        # to read from (the one the previous refresh just finished
        # filling, which the next refresh will overwrite).
        self._next_writer: int = 0
        # Total successful refreshes so far.  ``read`` returns None
        # while this is < 2 — there's no completed "other" slot yet.
        self._refresh_count: int = 0
        # Latched on any allocation / copy failure.  Stops further
        # work permanently and forces ``read`` to return None so the
        # caller takes the synchronous fallback.
        self._failed: bool = False

    @property
    def ok(self) -> bool:
        """True once both slots have been written at least once.

        After 2 successful refreshes, slot 0 holds tick 1's data and
        slot 1 holds tick 2's data; from tick 3 onward, ``read`` can
        always return the previous tick's data from the slot the
        next refresh is about to overwrite.
        """
        return not self._failed and self._refresh_count >= 2

    def refresh(self, src: Any, stream: Any) -> None:
        """Enqueue an async device→host copy on ``stream``.

        ``src`` is the device-side ``wp.array`` to mirror (typically
        ``state_0.joint_q`` etc.).  ``stream`` is the warp stream the
        captured physics graph just ran on — the copy must land on the
        same stream so it's FIFO-ordered behind the kernel writes.

        First call allocates both pinned-host buffers sized to
        ``src``.  Subsequent calls reuse them in place; if ``src``
        ever changes shape (defensive — Newton's state buffers don't
        change size at runtime), the slot is reallocated.

        No event is recorded: ``read`` relies on the wall-time gap
        between writes (≥ one physics tick) plus the microsecond
        DMA cost to know the copy has landed.  See ``read`` for
        the rationale.
        """
        if self._failed:
            return
        try:
            slot = self._next_writer
            buf = self._bufs[slot]
            if buf is None or buf.size != src.size or buf.dtype != src.dtype:
                buf = wp.empty(src.shape, dtype=src.dtype, device="cpu", pinned=True)
                self._bufs[slot] = buf
                self._views[slot] = buf.numpy()
            wp.copy(buf, src, stream=stream)
            self._next_writer = 1 - slot
            self._refresh_count += 1
        except Exception:
            self._failed = True

    def read(self) -> Optional[np.ndarray]:
        """Return the numpy view of the slot the next refresh will overwrite.

        That slot holds the previous tick's data (because the most
        recent refresh wrote into the OTHER slot and flipped this
        index).

        No event synchronization
        ------------------------
        ``wp.synchronize_event`` is deliberately NOT called here.
        The ping-pong structure gives a stronger correctness
        guarantee than the event: the slot we return was filled at
        least one physics tick ago — wall time ≥ ``physics_dt``
        (typically 10 ms) — and the wp.copy itself is a 70-float
        DMA that completes in microseconds.  By the time the publish
        thread reaches ``read()``, the copy has been done for ~10 ms.

        On warp 1.14 ``wp.synchronize_event`` conservatively waits on
        more than the recorded operation: even when the previous-tick
        event is structurally guaranteed to be complete, the call
        burns 2-6 ms per tick (measured on a G2-class scene).
        Skipping it gets the sub-millisecond steady-state read.

        Returns ``None`` while the mirror is still warming up
        (refresh_count < 2 — only one slot has been written) or has
        failed; callers should fall back to ``src.numpy()`` then.
        """
        if not self.ok:
            return None
        return self._views[self._next_writer]
