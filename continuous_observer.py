import asyncio
import json
import websockets
import ollama
import socketio
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_MODEL = "qwen2.5:32b"
WEBSOCKET_URI = "ws://localhost:8019"
HUB_URL = "http://localhost:8002"

SYSTEM_PROMPT = """You are a highly observant, sassy AI co-host watching a live gaming stream.
You receive a sensory snapshot every 3 seconds.

RULES FOR RESPONDING:
1. If the player speaks directly to you, or says something you can riff on, RESPOND.
2. If the player dies, gets jumpscared, or panics (e.g., saying "I'm screwed"), MOCK THEM or act shocked.
3. Keep responses strictly under 2 sentences. Speak naturally, as if on voice chat.
4. ESCAPE HATCH: If nothing highly entertaining, dangerous, or funny is happening, and the player is just focused or wandering, YOU MUST OUTPUT EXACTLY: <SILENCE>
Do not explain your silence. Just output <SILENCE>."""

class ContinuousObserver:
    def __init__(self):
        self.is_generating = False
        self.sio = socketio.AsyncClient()
        print(f"🧠 Initializing Qwen Observer ({OLLAMA_MODEL})...")

    async def connect(self):
        # Connect to the Hub to send responses back to the UI
        try:
            await self.sio.connect(HUB_URL)
            print("✅ Connected to Hub!")
        except Exception as e:
            print(f"⚠️ Could not connect to Hub: {e}")

        # Connect to the local WebSocket to receive sensory data
        while True:
            try:
                print(f"🔌 Connecting to {WEBSOCKET_URI}...")
                async for websocket in websockets.connect(WEBSOCKET_URI):
                    print("✅ Connected to Sensory Stream!")
                    async for message in websocket:
                        await self.handle_message(message)
            except websockets.exceptions.ConnectionClosed:
                print("⚠️ Connection lost. Retrying in 3s...")
                await asyncio.sleep(3)
            except ConnectionRefusedError:
                print("⚠️ Aggregator not running. Retrying in 3s...")
                await asyncio.sleep(3)

    async def handle_message(self, message: str):
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return

        if data.get("type") != "continuous_context":
            return

        if self.is_generating:
            return

        await self.evaluate_scene(data.get("context_string"))

    async def evaluate_scene(self, context_string: str):
        self.is_generating = True
        
        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: ollama.generate(
                    model=OLLAMA_MODEL,
                    prompt=f"{SYSTEM_PROMPT}\n\n{context_string}\n\nYOUR RESPONSE:",
                    options={"temperature": 0.7, "num_predict": 60}
                )
            )

            reply = response.get("response", "").strip()

            if reply == "<SILENCE>" or not reply:
                # Tell the UI we decided to stay silent
                if self.sio.connected:
                    await self.sio.emit("ai_response", {"text": "<SILENCE>", "timestamp": datetime.now().isoformat()})
            else:
                # The AI decided to speak!
                time_now = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{time_now}] 🎙️ AI: {reply}")
                
                # Send the spoken text back to the UI
                if self.sio.connected:
                    await self.sio.emit("ai_response", {"text": reply, "timestamp": datetime.now().isoformat()})
                
        except Exception as e:
            print(f"❌ Ollama Error: {e}")
        finally:
            self.is_generating = False

if __name__ == "__main__":
    observer = ContinuousObserver()
    asyncio.run(observer.connect())