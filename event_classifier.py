"""
EventClassifier for the Context Service.
Takes a fused snapshot of all three senses and returns a structured
event label. Deliberately uses a fast/cheap model — we want <300ms.
"""

import json
import time
from openai import OpenAI
from config import CLASSIFIER_MODEL, CLASSIFIER_MAX_TOKENS, OPENAI_API_KEY

# All possible event types the classifier can emit.
# Keep this list stable — the AI personality layer keys off these strings.
EVENT_TYPES = [
    "PLAYER_DEATH",
    "PLAYER_WIN",
    "PLAYER_RESPAWN",
    "TENSE_MOMENT",
    "JUMPSCARED",
    "FUNNY_MOMENT",
    "CONFUSION",          # Player seems lost / stuck
    "CHAT_INTERACTION",   # Chat/someone talking to the player
    "PLAYER_SPEAKING",    # User speaking directly (mic active)
    "CONVERSATION",       # Back-and-forth dialogue happening on screen
    "CUTSCENE",           # Non-interactive cinematic moment
    "MENU_SCREEN",        # Player is in a menu / loading screen
    "BORING_LULL",        # Nothing interesting happening
    "IDLE",               # No meaningful input from any sense
]

SYSTEM_PROMPT = """You are a real-time event classifier for a live gaming/streaming session.
You receive a fused snapshot from three senses: vision (screen), audio (desktop sound), and mic (player voice).
Each entry has a timestamp so you know the order of events.

Your job: classify what is CURRENTLY happening into exactly ONE event type.

Rules:
- Prefer the most specific event type that fits.
- If the player is speaking (mic has recent content), lean toward PLAYER_SPEAKING or CHAT_INTERACTION.
- IDLE means truly nothing from any sense in the window.
- BORING_LULL means senses are active but nothing noteworthy is happening.
- Output ONLY valid JSON. No explanation, no markdown.

Output format:
{"event": "EVENT_TYPE", "confidence": 0.00, "summary": "one short sentence"}
"""

USER_TEMPLATE = """CURRENT SENSE SNAPSHOT:

VISION (screen, newest first):
{vision}

DESKTOP AUDIO (newest first):
{audio}

MIC / PLAYER VOICE (newest first):
{mic}

Classify the current moment. Choose from: {event_types}"""


class EventClassifier:
    def __init__(self, debug_mode: bool = False):
        self.client     = OpenAI(api_key=OPENAI_API_KEY)
        self.debug_mode = debug_mode

    def classify(self, vision_lines: list[str], audio_lines: list[str], mic_lines: list[str]) -> dict | None:
        """
        Returns a dict like:
            {"event": "PLAYER_DEATH", "confidence": 0.91, "summary": "..."}
        or None on failure.
        """
        vision_text = "\n".join(vision_lines) if vision_lines else "(no recent vision data)"
        audio_text  = "\n".join(audio_lines)  if audio_lines  else "(no recent audio data)"
        mic_text    = "\n".join(mic_lines)     if mic_lines    else "(silence)"

        user_msg = USER_TEMPLATE.format(
            vision=vision_text,
            audio=audio_text,
            mic=mic_text,
            event_types=", ".join(EVENT_TYPES),
        )

        try:
            t0       = time.time()
            response = self.client.chat.completions.create(
                model    = CLASSIFIER_MODEL,
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                max_tokens  = CLASSIFIER_MAX_TOKENS,
                temperature = 0.1,
            )
            elapsed = time.time() - t0

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if the model misbehaves
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            result = json.loads(raw)

            # Validate
            if result.get("event") not in EVENT_TYPES:
                result["event"] = "IDLE"
            result["confidence"] = float(result.get("confidence", 0.0))
            result["summary"]    = str(result.get("summary", ""))
            result["latency_ms"] = round(elapsed * 1000)

            if self.debug_mode:
                print(f"[Classifier] {result['event']} ({result['confidence']:.2f}) in {result['latency_ms']}ms — {result['summary']}")

            return result

        except json.JSONDecodeError as e:
            print(f"[Classifier] JSON parse error: {e} | raw: {repr(raw)}")
            return None
        except Exception as e:
            print(f"[Classifier] Error: {e}")
            return None