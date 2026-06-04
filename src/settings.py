from __future__ import annotations

import os

from dynaconf import Dynaconf


settings = Dynaconf(
	settings_files=["settings.yaml"],
	load_dotenv=True,
)
