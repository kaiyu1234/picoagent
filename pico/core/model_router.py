"""Model client routing for capability-specific calls."""

from __future__ import annotations


class ModelClientRouter:
    def __init__(self, main_client, vision_client=None, vision_client_factory=None):
        self.main_client = main_client
        self._vision_client = vision_client
        self._vision_client_factory = vision_client_factory

    def default_client(self):
        return self.main_client

    def replace_default_client(self, previous, replacement):
        if self.main_client is previous:
            self.main_client = replacement
        if self._vision_client is previous:
            self._vision_client = replacement

    def vision_client(self):
        if self._vision_client is not None:
            return self._vision_client
        if self._vision_client_factory is None:
            return self.main_client
        self._vision_client = self._vision_client_factory()
        return self._vision_client

    def client_for_input(self, model_input):
        if getattr(model_input, "image_count", 0):
            return self.vision_client()
        return self.main_client
