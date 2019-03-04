# -*- coding: utf-8 -*-

import os
import sys
import yaml
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

class Config:

  def __init__(self, project_dir = os.getenv("PROJECT_DIR"), mode = os.getenv("MODE")):

    self.project_dir = project_dir
    self.mode = mode

    with open(os.path.join(self.project_dir, 'config', 'config_%s.yaml' % (self.mode)), 'r') as ymlfile:
      self.config = yaml.load(ymlfile)

  def get(self, key):
    return self.config[key]

