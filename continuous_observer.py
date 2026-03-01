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

SYSTEM_PROMPT = """You are a passive, analytical AI observer monitoring a live gaming stream. 
You receive raw sensory snapshots containing vision (screen contents), audio (desktop sounds), and mic (streamer's voice).

YOUR GOAL: 
Process this raw data and provide a unified, highly detailed, objective analysis of the current state of the stream. 

RULES:
1. DO NOT speak to the streamer or act like a chatbot. 
2. Synthesize the inputs. Instead of listing "vision says this, mic says this," weave them into a coherent paragraph (e.g., "The streamer is looking at X while saying Y").
3. Capture the emotional tone, the specific game state, and any chat interactions occurring.
4. ESCAPE HATCH: If the current snapshot is identical in meaning to the previous few seconds (e.g., just standing in a hallway in silence), output EXACTLY: <SILENCE>"""

class ContinuousObserver:
    def __init__(self):
        self.is_generating = False
        self.sio = socketio.AsyncClient()
        print(f"🧠 Initializing Qwen Observer ({OLLAMA_MODEL})...")

    async def connect(self):
        try:
            await self.sio.connect(HUB_URL)
            print("✅ Connected to Hub!")
        except Exception as e:
            print(f"⚠️ Could not connect to Hub: {e}")

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
                    prompt=f"{SYSTEM_PROMPT}\n\n{context_string}\n\nANALYSIS:",
                    options={
                        "temperature": 0.3,   # Lower temperature for objective analysis
                        "num_predict": 150    # Give it enough tokens to write a detailed summary
                    }
                )
            )

            reply = response.get("response", "").strip()

            if reply == "<SILENCE>" or not reply:
                if self.sio.connected:
                    await self.sio.emit("ai_response", {"text": "<SILENCE>", "timestamp": datetime.now().isoformat()})
            else:
                time_now = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{time_now}] 📊 ANALYSIS: {reply}")
                
                if self.sio.connected:
                    await self.sio.emit("ai_response", {"text": reply, "timestamp": datetime.now().isoformat()})
                
        except Exception as e:
            print(f"❌ Ollama Error: {e}")
        finally:
            self.is_generating = False

if __name__ == "__main__":
    observer = ContinuousObserver()
    asyncio.run(observer.connect())