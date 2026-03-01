"""
ContextService — aggregates all three sense streams, timestamps them,
deduplicates noise, classifies events, and emits clean context packets
to the Hub for the AI personality layer to consume.
"""
import asyncio
import threading
import time
import traceback
from datetime import datetime, timezone

import socketio

from config import (
    AUDIO_BUFFER_SIZE, AUDIO_STALE_S,
    CONFIDENCE_THRESHOLD, EVENT_COOLDOWN_S,
    FUSION_WINDOW_S, HUB_URL,
    MIC_BUFFER_SIZE, MIC_STALE_S,
    SERVICE_NAME,
    VISION_BUFFER_SIZE, VISION_STALE_S,
)
from event_classifier import EventClassifier
from sense_buffer import SenseBuffer
from websocket_server import WebSocketServer


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [{SERVICE_NAME}] {msg}", flush=True)


class ContextService:
    def __init__(self):
        log("🧠 Initializing …")
        self._shutting_down  = False
        self._shutdown_lock  = threading.Lock()
        self._hub_emit_count = 0
        self._event_count    = 0
        self._classify_count = 0

        # ── Sense buffers ─────────────────────────────────────────────────────
        self.vision_buf = SenseBuffer("vision", maxlen=VISION_BUFFER_SIZE, stale_after_s=VISION_STALE_S)
        self.audio_buf  = SenseBuffer("audio",  maxlen=AUDIO_BUFFER_SIZE,  stale_after_s=AUDIO_STALE_S)
        self.mic_buf    = SenseBuffer("mic",     maxlen=MIC_BUFFER_SIZE,    stale_after_s=MIC_STALE_S)

        # ── Event dedup state ─────────────────────────────────────────────────
        self._last_event_times: dict[str, float] = {}
        self._last_classified_state = ""

        # ── Classifier ────────────────────────────────────────────────────────
        log("Initializing EventClassifier …")
        try:
            self.classifier = EventClassifier(debug_mode=True)
            log("✅ EventClassifier ready")
        except Exception as e:
            log(f"❌ EventClassifier FAILED: {e}")
            raise

        # ── WebSocket server ──────────────────────────────────────────────────
        log("Creating WebSocket broadcast server …")
        self.ws_server = WebSocketServer()

        # ── Hub client ────────────────────────────────────────────────────────
        self.sio      = socketio.AsyncClient(reconnection=True, reconnection_delay=5)
        self.hub_loop: asyncio.AbstractEventLoop | None = None
        self._register_hub_events()

        # ── Fusion loop control ───────────────────────────────────────────────
        self._fusion_interval_s = FUSION_WINDOW_S / 2
        self._fusion_thread: threading.Thread | None = None

        log("✅ ContextService initialized")

    def run(self):
        self.hub_loop = asyncio.new_event_loop()
        threading.Thread(target=self._hub_thread, args=(self.hub_loop,), daemon=True, name="CtxHub").start()
        log("Hub thread started")

        self.ws_server.start()
        log("WebSocket server started")

        self._fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True, name="CtxFusion")
        self._fusion_thread.start()
        log(f"Fusion loop started (interval={self._fusion_interval_s:.1f}s)")

        log("✅ Running — listening for sense data …")
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
        log(f"🛑 Shutting down (hub emits: {self._hub_emit_count}, events: {self._event_count}, classifies: {self._classify_count})")
        self.ws_server.stop()
        if self.hub_loop:
            self.hub_loop.call_soon_threadsafe(self.hub_loop.stop)
        log("🛑 Stopped.")

    # ── Hub event registration ─────────────────────────────────────────────────

    def _register_hub_events(self):

        @self.sio.event
        async def connect():
            log(f"✅ Hub CONNECTED → {HUB_URL}")

        @self.sio.event
        async def disconnect():
            log("⚠️  Hub DISCONNECTED")

        @self.sio.event
        async def connect_error(data):
            log(f"❌ Hub CONNECTION ERROR: {data}")

        # ── Vision ────────────────────────────────────────────────────────────
        @self.sio.on("vision_context")
        async def on_vision(data):
            ctx = data.get("context", "").strip()
            ts  = data.get("timestamp")
            if ctx:
                self.vision_buf.add(ctx, timestamp=ts)
                log(f"👁️  vision: {repr(ctx[:80])}")

        @self.sio.on("text_update")
        async def on_text_update(data):
            ctx = data.get("content", "").strip()
            ts  = data.get("timestamp")
            if ctx:
                self.vision_buf.add(ctx, timestamp=ts)

        # ── Desktop audio ─────────────────────────────────────────────────────
        @self.sio.on("audio_context")
        async def on_audio(data):
            ctx = data.get("context", "").strip()
            src = data.get("metadata", {}).get("source", "audio")
            ts  = data.get("timestamp")
            if ctx and src != "microphone":
                self.audio_buf.add(ctx, timestamp=ts)
                log(f"🔊 audio: {repr(ctx[:80])}")

        # ── Microphone ────────────────────────────────────────────────────────
        @self.sio.on("spoken_word_context")
        async def on_mic(data):
            ctx = data.get("context", "").strip()
            ts  = data.get("timestamp")
            if ctx:
                self.mic_buf.add(ctx, timestamp=ts)
                log(f"🎤 mic: {repr(ctx[:80])}")

        @self.sio.on("transcript_enriched")
        async def on_enriched(data):
            text = data.get("text", "").strip()
            src  = data.get("speaker", "unknown")
            ts   = data.get("timestamp")
            if text:
                self.audio_buf.add(f"[{src}] {text}", timestamp=ts)

    # ── Hub helpers ───────────────────────────────────────────────────────────

    def _hub_thread(self, loop: asyncio.AbstractEventLoop):
        asyncio.set_event_loop(loop)
        loop.create_task(self._hub_connection_loop())
        loop.run_forever()

    async def _hub_connection_loop(self):
        while not self._shutting_down:
            if not self.sio.connected:
                try:
                    log(f"Attempting hub connect → {HUB_URL} …")
                    await self.sio.connect(HUB_URL)
                except Exception as e:
                    log(f"⚠️  Hub connect failed: {e} — retry in 5s")
                    await asyncio.sleep(5)
            await asyncio.sleep(2)

    def _emit_to_hub(self, event: str, data: dict):
        if not self.sio.connected or not self.hub_loop:
            log(f"⚠️  SKIPPED hub emit: {event}")
            return
        try:
            asyncio.run_coroutine_threadsafe(self.sio.emit(event, data), self.hub_loop)
            self._hub_emit_count += 1
            log(f"→ HUB [{event}] {str(data)[:160]}")
        except Exception as e:
            log(f"❌ HUB EMIT ERROR: {e}")
            log(traceback.format_exc())

    # ── Fusion loop ───────────────────────────────────────────────────────────

    def _fusion_loop(self):
        log("🔄 Fusion loop active")
        while not self._shutting_down:
            t0 = time.time()
            try:
                self._run_fusion_tick()
            except Exception as e:
                log(f"❌ Fusion tick error: {e}")
                log(traceback.format_exc())
            elapsed = time.time() - t0
            sleep   = max(0, self._fusion_interval_s - elapsed)
            time.sleep(sleep)

    def _run_fusion_tick(self):
        vision_lines = self.vision_buf.formatted_lines()
        audio_lines  = self.audio_buf.formatted_lines()
        mic_lines    = self.mic_buf.formatted_lines()

        if not vision_lines and not audio_lines and not mic_lines:
            return

        current_state = str(vision_lines) + str(audio_lines) + str(mic_lines)
        if current_state == self._last_classified_state:
            return 
        
        self._last_classified_state = current_state

        self._classify_count += 1
        result = self.classifier.classify(vision_lines, audio_lines, mic_lines)

        if result is None:
            return

        event_type = result["event"]
        confidence = result["confidence"]
        summary    = result["summary"]

        if confidence < CONFIDENCE_THRESHOLD:
            log(f"  ↳ Low confidence ({confidence:.2f}) for {event_type} — skipping")
            return

        is_conversational = event_type in ["PLAYER_SPEAKING", "CHAT_INTERACTION"]
        last_emit = self._last_event_times.get(event_type, 0)
        
        if not is_conversational and (time.time() - last_emit < EVENT_COOLDOWN_S):
            log(f"  ↳ {event_type} on cooldown ({time.time() - last_emit:.1f}s / {EVENT_COOLDOWN_S}s)")
            return

        self._last_event_times[event_type] = time.time()
        self._event_count += 1

        now_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")

        packet = {
            "event":      event_type,
            "confidence": round(confidence, 3),
            "summary":    summary,
            "timestamp":  now_iso,
            "context": {
                "vision": vision_lines,
                "audio":  audio_lines,
                "mic":    mic_lines,
            },
            "player_speaking": bool(mic_lines and not self.mic_buf.is_stale()),
            "has_vision":      bool(vision_lines),
            "has_audio":       bool(audio_lines),
        }

        log(f"🎯 EVENT #{self._event_count}: {event_type} ({confidence:.2f}) — {summary}")

        self.ws_server.broadcast({"type": "classified_event", **packet})
        self._emit_to_hub("classified_event", packet)

        readable = self._build_readable_context(packet)
        self._emit_to_hub("ai_context", {
            "context":   readable,
            "event":     event_type,
            "timestamp": now_iso,
        })
        self.ws_server.broadcast({
            "type":      "ai_context",
            "context":   readable,
            "event":     event_type,
            "timestamp": now_iso,
        })

        self.vision_buf.clear()
        self.audio_buf.clear()
        self.mic_buf.clear()

    def _build_readable_context(self, packet: dict) -> str:
        ts   = packet["timestamp"][11:19]
        lines = [
            f"[{ts}] EVENT: {packet['event']} (conf {packet['confidence']:.2f})",
            f"What's happening: {packet['summary']}",
        ]

        vision = packet["context"]["vision"]
        audio  = packet["context"]["audio"]
        mic    = packet["context"]["mic"]

        if vision:
            lines.append(f"Screen: {vision[0]}")
        if audio:
            lines.append(f"Audio:  {audio[0]}")
        lines.append(f"Player: {mic[0] if mic else '(silent)'}")

        return "\n".join(lines)