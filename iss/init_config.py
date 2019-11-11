import os
from dotenv import find_dotenv, load_dotenv

from iss.tools import Config

load_dotenv(find_dotenv())
CONFIG = Config(project_dir = os.getenv("PROJECT_DIR"), mode = os.getenv("MODE"))

