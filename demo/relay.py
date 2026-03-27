#!/usr/bin/env python3
"""Standalone NIP-01 relay for local nostrain demos.

Usage: python demo/relay.py [PORT]
Default port: 7777
"""

import asyncio
import json
import signal
import sys
import time

import websockets


def _frame(data):
    return json.dumps(data, separators=(",", ":"))


class DemoRelay:
    def __init__(self, port: int = 7777) -> None:
        self.port = port
        self._events: list[dict] = []
        self._subscriptions: list[dict] = []
        self._stats = {"events_stored": 0, "connections": 0, "publishes": 0}

    async def run(self) -> None:
        stop = asyncio.Event()
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)

        async with websockets.serve(self._handle, "127.0.0.1", self.port):
            url = f"ws://127.0.0.1:{self.port}"
            print(f"\033[1;36m{'=' * 60}\033[0m")
            print("\033[1;36m  NOSTRAIN RELAY\033[0m")
            print(f"\033[1;36m  {url}\033[0m")
            print(f"\033[1;36m{'=' * 60}\033[0m")
            print()
            await stop.wait()
            print("\n\033[1;33mRelay shutting down.\033[0m")

    async def _handle(self, ws) -> None:
        self._stats["connections"] += 1
        remote = ws.remote_address
        self._log("CONN", f"New connection from {remote[0]}:{remote[1]}")
        try:
            async for raw in ws:
                frame = json.loads(raw)
                msg_type = frame[0]
                if msg_type == "EVENT":
                    await self._on_publish(ws, frame)
                elif msg_type == "REQ":
                    await self._on_subscribe(ws, frame)
                elif msg_type == "CLOSE":
                    await self._on_close(ws, frame)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._subscriptions = [s for s in self._subscriptions if s["ws"] is not ws]
            self._log("DISC", f"Disconnected {remote[0]}:{remote[1]}")

    async def _on_publish(self, ws, frame) -> None:
        event = frame[1]
        event_id = event.get("id", "?")[:12]
        kind = event.get("kind", "?")
        pubkey = event.get("pubkey", "?")[:12]
        self._stats["publishes"] += 1
        self._stats["events_stored"] += 1
        self._events.append(event)

        kind_label = {33333: "GRADIENT", 33334: "HEARTBEAT", 33335: "CHECKPOINT"}.get(
            kind, f"KIND:{kind}"
        )
        tags = {t[0]: t[1] if len(t) > 1 else "" for t in event.get("tags", [])}
        round_num = tags.get("round", "?")
        run_name = tags.get("run", "?")

        color = {33333: "32", 33334: "35", 33335: "33"}.get(kind, "37")
        self._log(
            kind_label,
            f"run=\033[1m{run_name}\033[0m round=\033[1m{round_num}\033[0m "
            f"pubkey={pubkey}.. id={event_id}..",
            color=color,
        )

        await ws.send(_frame(["OK", event.get("id", ""), True, "stored"]))
        await self._broadcast(event)

    async def _on_subscribe(self, ws, frame) -> None:
        sub_id = str(frame[1])
        filters = tuple(frame[2:])
        self._subscriptions.append({"ws": ws, "sub_id": sub_id, "filters": filters})

        matched = 0
        for ev in self._events:
            if any(self._match(ev, f) for f in filters):
                await ws.send(_frame(["EVENT", sub_id, ev]))
                matched += 1
        await ws.send(_frame(["EOSE", sub_id]))
        self._log("SUB", f"sub={sub_id[:8]}.. matched={matched} stored", color="34")

    async def _on_close(self, ws, frame) -> None:
        sub_id = str(frame[1])
        self._subscriptions = [
            s
            for s in self._subscriptions
            if not (s["ws"] is ws and s["sub_id"] == sub_id)
        ]
        try:
            await ws.send(_frame(["CLOSED", sub_id, "closed"]))
        except websockets.ConnectionClosed:
            pass

    async def _broadcast(self, event) -> None:
        stale = []
        for sub in self._subscriptions:
            if not any(self._match(event, f) for f in sub["filters"]):
                continue
            try:
                await sub["ws"].send(_frame(["EVENT", sub["sub_id"], event]))
            except Exception:
                stale.append(id(sub))
        if stale:
            self._subscriptions = [s for s in self._subscriptions if id(s) not in stale]

    def _match(self, event, filt) -> bool:
        if "kinds" in filt and int(event.get("kind", 0)) not in filt["kinds"]:
            return False
        if "since" in filt and int(event.get("created_at", 0)) < int(filt["since"]):
            return False
        if "until" in filt and int(event.get("created_at", 0)) > int(filt["until"]):
            return False
        if "authors" in filt:
            if str(event.get("pubkey", "")) not in {str(v) for v in filt["authors"]}:
                return False
        for key, values in filt.items():
            if not key.startswith("#"):
                continue
            tag_name = key[1:]
            tag_vals = {
                str(t[1])
                for t in event.get("tags", [])
                if isinstance(t, list) and len(t) >= 2 and t[0] == tag_name
            }
            if not tag_vals.intersection({str(v) for v in values}):
                return False
        return True

    def _log(self, label, msg, color="37") -> None:
        ts = time.strftime("%H:%M:%S")
        total = self._stats["events_stored"]
        print(
            f"\033[90m{ts}\033[0m "
            f"[\033[1;{color}m{label:>10}\033[0m] "
            f"{msg} "
            f"\033[90m(total: {total})\033[0m"
        )


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7777
    asyncio.run(DemoRelay(port).run())
