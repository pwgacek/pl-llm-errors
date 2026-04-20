from __future__ import annotations

import os

from dynaconf import Dynaconf


settings = Dynaconf(
	settings_files=["settings.yaml"],
	environments=True,
	env_switcher="ENV_FOR_DYNACONF",
	default_env="default",
	env="local",
	merge_enabled=True,
	load_dotenv=True,
)
