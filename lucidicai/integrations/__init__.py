"""Third-party integrations for Lucidic AI SDK.

This module provides integrations with external platforms and frameworks
that have their own OpenTelemetry instrumentation.
"""

from .livekit import setup_livekit, LucidicLiveKitExporter

__all__ = ["setup_livekit", "LucidicLiveKitExporter"]
