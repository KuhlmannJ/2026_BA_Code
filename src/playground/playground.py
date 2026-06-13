from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from openai import OpenAI

client = OpenAI(
    base_url="https://gpt.uni-muenster.de/v1"
)

completion = client.chat.completions.create(
    model="mistral-small",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ]
)

print(completion.choices[0].message.content)