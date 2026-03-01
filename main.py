#!/usr/bin/env python3
"""
Context Service — Entry Point
==============================
Aggregates vision, desktop audio, and microphone streams from the Hub.
Timestamps all events, deduplicates noise, classifies moments into named
events, and emits clean structured context for the AI personality layer.

WebSocket clients:  ws://localhost:8019
Health check:       GET  http://localhost:8020/health
Shutdown:           POST http://localhost:8020/shutdown

Hub events consumed:
    vision_context, text_update        ← from vision_service
    audio_context, transcript_enriched ← from stream_audio_service
    spoken_word_context                ← from microphone_audio_service

Hub events emitted:
    classified_event   ← structured event packet with full sense context
    ai_context         ← single readable string for direct AI consumption
"""

import threading
import subprocess
import os
import signal
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_THIS_DIR)

import http_control
from service import ContextService

_service: ContextService | None = None
_observer_proc: subprocess.Popen | None = None


def _run_observer():
    global _observer_proc
    _observer_proc = subprocess.Popen(
        [sys.executable, os.path.join(_THIS_DIR, "continuous_observer.py")],
        cwd=_THIS_DIR
    )
    _observer_proc.wait()


def _shutdown(*_):
    global _service, _observer_proc

    if _observer_proc and _observer_proc.poll() is None:
        print("🛑 Terminating observer subprocess...")
        _observer_proc.terminate()
        try:
            _observer_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("⚠️  Observer didn't exit cleanly — killing.")
            _observer_proc.kill()

    if _service:
        _service.stop()

    sys.exit(0)


def main():
    global _service
    _service = ContextService()

    http_control.start(shutdown_callback=_shutdown)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    observer_thread = threading.Thread(target=_run_observer, daemon=True, name="Observer")
    observer_thread.start()

    _service.run()


if __name__ == "__main__":
    main()