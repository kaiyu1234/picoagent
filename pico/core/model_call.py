"""Interruptible model-call adapter for the synchronous runtime engine."""

import threading

from ..providers.base import complete_model


class ModelCallAborted(RuntimeError):
    pass


def complete_model_interruptibly(agent, prompt, max_new_tokens, **kwargs):
    client = agent.model_client
    completed = threading.Event()
    outcome = {}

    def invoke():
        try:
            outcome["result"] = complete_model(
                client, prompt, max_new_tokens, **kwargs
            )
        except Exception as exc:  # noqa: BLE001 - forwarded to the engine thread
            outcome["error"] = exc
        finally:
            completed.set()

    threading.Thread(
        target=invoke,
        daemon=True,
        name="pico-model-call",
    ).start()
    while not completed.wait(0.05):
        if agent.abort_requested:
            agent.detach_aborted_model_client(client)
            raise ModelCallAborted("model call aborted")

    if agent.abort_requested:
        agent.detach_aborted_model_client(client)
        raise ModelCallAborted("model call aborted")
    error = outcome.get("error")
    if error is not None:
        raise error
    return outcome["result"]
