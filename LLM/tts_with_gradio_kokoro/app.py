import os
from fastrtc import (ReplyOnPause, Stream, get_stt_model, get_tts_model)
from openai import OpenAI
import json
from huggingface_hub import login
import httpx
import requests

s = requests.Session()
s.verify = False

with open('./config.json', 'r') as config_file:
    config_dict = json.load(config_file)

# Extract API URLs and API Keys from the configuration
sambanova_api_key = config_dict.get("SAMBANOVA_API_KEY")
openai_api_key = config_dict.get("OPENAI_API_KEY")

login("")
#export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/ssl/certs/ca-certificates.crt"

sambanova_client = OpenAI(
  api_key= sambanova_api_key, 
  base_url="https://api.sambanova.ai/v1",
  http_client = httpx.Client(verify=False)
)

stt_model = get_stt_model()
tts_model = get_tts_model()

def echo(audio):
  prompt = stt_model.stt(audio)

  response = sambanova_client.chat.completions.create(
    model="Meta-Llama-3.2-3B-Instruct",
    messages=[{"role":"user", "content": prompt}],
    max_tokens = 200,
  )

  prompt = response.choices[0].message.content
  print(response.choices)

  for audio_chunk in tts_model.stream_tts_sync(prompt):
    yield audio_chunk

stream = Stream(ReplyOnPause(echo), modality = "audio", mode = "send-receive")
stream.ui.launch()