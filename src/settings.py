from __future__ import annotations

import os

from dynaconf import Dynaconf


settings = Dynaconf(
	settings_files=["settings.yaml"],
	environments=False,
	load_dotenv=True,
)


if not settings.get("evaluation.api_key"):
	env_api_key = os.getenv("API_KEY")
	if env_api_key:
		settings.set("evaluation.api_key", env_api_key)