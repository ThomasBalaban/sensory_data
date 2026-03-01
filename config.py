"""
Configuration for the Context Aggregator Service.
Subscribes to all three sense services via Hub, cleans/timestamps/fuses
events, and emits structured context packets for the AI personality layer.
"""
from api_keys import GEMINI_API_KEY

# ── Fusion window ─────────────────────────────────────────────────────────────
# How many seconds of events to consider "the same moment"
FUSION_WINDOW_S = 4.0

# How long each sense's last value stays valid before it's considered stale
VISION_STALE_S       = 8.0
AUDIO_STALE_S        = 6.0
MIC_STALE_S          = 10.0

# Rolling buffer sizes (number of items kept per sense)
VISION_BUFFER_SIZE   = 5
AUDIO_BUFFER_SIZE    = 8
MIC_BUFFER_SIZE      = 4

# ── Classifier ────────────────────────────────────────────────────────────────
# Model used for event classification (fast + cheap)
CLASSIFIER_MODEL      = "gemini-2.0-flash"
CLASSIFIER_MAX_TOKENS = 120

# Only emit a classified_event if confidence >= this
CONFIDENCE_THRESHOLD = 0.60

# Minimum seconds between emitting the same event type
EVENT_COOLDOWN_S     = 8.0

# ── Network ───────────────────────────────────────────────────────────────────
WEBSOCKET_PORT    = 8019
HTTP_CONTROL_PORT = 8020
HUB_URL           = "http://localhost:8002"

SERVICE_NAME = "context_service"