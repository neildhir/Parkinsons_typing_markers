import yaml

import yaml
from box import Box

with open("config/cfg.yml", "r") as ymlfile:
  cfg = Box(yaml.safe_load(ymlfile))

