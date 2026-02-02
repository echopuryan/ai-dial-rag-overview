import os
from dotenv import load_dotenv

# Load variables from .env file into environment
load_dotenv()

DIAL_URL = 'https://ai-proxy.lab.epam.com'
API_KEY = os.getenv('DIAL_API_KEY', '')