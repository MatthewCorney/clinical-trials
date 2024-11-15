from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from openai import OpenAI
import logging

load_dotenv()


class Settings(BaseSettings):
    openai_api_key: str = ''
    logging_level: str = 'INFO'
    model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "text-embedding-3-small"

    class Config:
        # Tell pydantic-settings to look for the .env file
        env_file = "../.env"
        env_file_encoding = "utf-8"


class CustomLogger(logging.Logger):
    def __init__(self, name, level="INFO"):
        """

        :param name:
        :param level:
        """
        super().__init__(name, level)

        formatter = logging.Formatter('%(asctime)s - %(filename)s - '
                                      '%(funcName)s - %(levelname)s: '
                                      '%(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')

        # Create a console handler and set the formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.addHandler(console_handler)


settings = Settings()
logger = CustomLogger(__name__, settings.logging_level)
client = OpenAI(api_key=settings.openai_api_key)
