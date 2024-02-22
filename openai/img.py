from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# 1. 用dall-e-3绘图
# prompt = "Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils"
# response = client.images.generate(
#     model="dall-e-3",
#     prompt=prompt,
#     size="1024x1024",
#     quality="standard",
#     n=1,
# )
# image_url = response.data[0].url
# print(image_url)

# 2. 用gpt-4-vision-preview识图
response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "What’s in this image?"},
        {
          "type": "image_url",
          "image_url": {
            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
          },
        },
      ],
    }
  ],
  max_tokens=300,
)

print(response)