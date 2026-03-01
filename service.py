"""
ContextService — Pure Sensory Aggregator.
Fuses vision, audio, and mic streams into a single snapshot every few seconds
and broadcasts it for the continuous AI observer to evaluate.
"""
import asyncio
import threading
import time
import traceback
from datetime import datetime, timezone

import socketio

from config import (
    AUDIO_BUFFER_SIZE, AUDIO_STALE_S,
    HUB_URL, MIC_BUFFER_SIZE, MIC_STALE_S,
    SERVICE_NAME, VISION_BUFFER_SIZE, VISION_STALE_S,
)
from sense_buffer import SenseBuffer
from websocket_server import WebSocketServer


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [{SERVICE_NAME}] {msg}", flush=True)


class ContextService:
    def __init__(self):
        log("🧠 Initializing Pure Aggregator …")
        self._shutting_down  = False
        self._shutdown_lock  = threading.Lock()
        self._emit_count     = 0

        # ── Sense buffers ─────────────────────────────────────────────────────
        self.vision_buf = SenseBuffer("vision", maxlen=VISION_BUFFER_SIZE, stale_after_s=VISION_STALE_S)
        self.audio_buf  = SenseBuffer("audio",  maxlen=AUDIO_BUFFER_SIZE,  stale_after_s=AUDIO_STALE_S)
        self.mic_buf    = SenseBuffer("mic",    maxlen=MIC_BUFFER_SIZE,    stale_after_s=MIC_STALE_S)

        self._last_emitted_state = ""

        # ── WebSocket server ──────────────────────────────────────────────────
        self.ws_server = WebSocketServer()

        # ── Hub client ────────────────────────────────────────────────────────
        self.sio      = socketio.AsyncClient(reconnection=True, reconnection_delay=5)
        self.hub_loop: asyncio.AbstractEventLoop | None = None
        self._register_hub_events()

        # ── Fusion loop control ───────────────────────────────────────────────
        self._fusion_interval_s = 3.0  # Force a snapshot every 3 seconds
        self._fusion_thread: threading.Thread | None = None

        log("✅ ContextService initialized")

    def run(self):
        self.hub_loop = asyncio.new_event_loop()
        threading.Thread(target=self._hub_thread, args=(self.hub_loop,), daemon=True, name="CtxHub").start()
        
        self.ws_server.start()
        
        self._fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True, name="CtxFusion")
        self._fusion_thread.start()
        
        log(f"🔄 Continuous fusion loop started (interval={self._fusion_interval_s:.1f}s)")
        try:
            while not self._shutting_down:
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        with self._shutdown_lock:
            if self._shutting_down:
                return
            self._shutting_down = True
        log("🛑 Shutting down...")
        self.ws_server.stop()
        if self.hub_loop:
            self.hub_loop.call_soon_threadsafe(self.hub_loop.stop)
        log("🛑 Stopped.")

    def _emit_to_hub(self, event: str, data: dict):
        if not self.sio.connected or not self.hub_loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.sio.emit(event, data), self.hub_loop)
        except Exception as e:
            log(f"❌ HUB EMIT ERROR: {e}")

    def _register_hub_events(self):
        @self.sio.event
        async def connect(): log(f"✅ Hub CONNECTED → {HUB_URL}")

        @self.sio.on("vision_context")
        async def on_vision(data):
            ctx, ts = data.get("context", "").strip(), data.get("timestamp")
            if ctx: self.vision_buf.add(ctx, timestamp=ts)

        @self.sio.on("audio_context")
        async def on_audio(data):
            ctx, src, ts = data.get("context", "").strip(), data.get("metadata", {}).get("source", "audio"), data.get("timestamp")
            if ctx and src != "microphone": self.audio_buf.add(ctx, timestamp=ts)

        @self.sio.on("spoken_word_context")
        async def on_mic(data):
            ctx, ts = data.get("context", "").strip(), data.get("timestamp")
            if ctx: self.mic_buf.add(ctx, timestamp=ts)

        @self.sio.on("transcript_enriched")
        async def on_enriched(data):
            text, src, ts = data.get("text", "").strip(), data.get("speaker", "unknown"), data.get("timestamp")
            if text: self.audio_buf.add(f"[{src}] {text}", timestamp=ts)

    def _hub_thread(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.create_task(self._hub_connection_loop())
        loop.run_forever()

    async def _hub_connection_loop(self):
        while not self._shutting_down:
            if not self.sio.connected:
                try: await self.sio.connect(HUB_URL)
                except Exception: await asyncio.sleep(5)
            await asyncio.sleep(2)

    def _fusion_loop(self):
        while not self._shutting_down:
            t0 = time.time()
            try: self._run_fusion_tick()
            except Exception as e: log(f"❌ Fusion tick error: {e}")
            elapsed = time.time() - t0
            time.sleep(max(0, self._fusion_interval_s - elapsed))

    def _run_fusion_tick(self):
        vision_lines = self.vision_buf.formatted_lines()
        
        # Keep the AI grounded in the environment even if the vision frame is technically "stale"
        if not vision_lines and self.vision_buf.latest():
            vision_lines = [self.vision_buf.latest().formatted()]

        audio_lines = self.audio_buf.formatted_lines()
        mic_lines   = self.mic_buf.formatted_lines()

        if not vision_lines and not audio_lines and not mic_lines:
            return

        current_state = str(vision_lines) + str(audio_lines) + str(mic_lines)
        if current_state == self._last_emitted_state:
            return 
        self._last_emitted_state = current_state

        now_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        
        # Build a single readable string for the LLM
        context_str = "CURRENT SENSE SNAPSHOT:\n\n"
        context_str += "VISION:\n" + ("\n".join(vision_lines) if vision_lines else "(no vision)") + "\n\n"
        context_str += "AUDIO:\n" + ("\n".join(audio_lines) if audio_lines else "(no audio)") + "\n\n"
        context_str += "MIC:\n" + ("\n".join(mic_lines) if mic_lines else "(silence)")

        packet = {
            "type": "continuous_context",
            "context_string": context_str,
            "timestamp": now_iso
        }

        log(f"📡 Emitting context tick (vision: {len(vision_lines)}, audio: {len(audio_lines)}, mic: {len(mic_lines)})")

        self.ws_server.broadcast(packet)
        self._emit_to_hub("continuous_context", packet)
        self._emit_count += 1