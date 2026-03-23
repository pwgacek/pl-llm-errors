from __future__ import annotations

from dynaconf import Dynaconf


settings = Dynaconf(
	settings_files=["settings.yaml"],
	environments=False,
	load_dotenv=False,
)