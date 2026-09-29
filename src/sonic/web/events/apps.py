import sys
import threading

from django.apps import AppConfig

from sonic.dataset.config import setting


class EventsConfig(AppConfig):
    name = "sonic.web.events"
    label = "events"
    verbose_name = "Audio and sound events"

    def ready(self) -> None:
        # Loading TensorFlow and the models takes several seconds; do it in the
        # background when the web server starts, not during the first upload.
        serving = "runserver" in sys.argv or setting("SONIC_PRELOAD_MODELS") == "1"
        if serving:
            from sonic.inference.python_model import get_classifier

            threading.Thread(target=get_classifier().warm_up, name="model-warm-up", daemon=True).start()
