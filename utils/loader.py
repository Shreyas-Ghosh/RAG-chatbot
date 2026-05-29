import base64
import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader


def encode_image(file) -> str:
    file.seek(0)
    return base64.b64encode(file.read()).decode("utf-8")


def extract_text(file, client) -> str:
    if file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text

    elif file.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        image_data = encode_image(file)
        extension = file.name.rsplit(".", 1)[-1].lower()
        media_type = "jpeg" if extension in ("jpg", "jpeg") else extension

        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{media_type};base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": """Describe this image in as much detail as possible.
Include: main subjects, colors, objects, setting, mood, any visible text,
and anything else notable. Be thorough — this description will be used
to answer questions about the image."""
                        }
                    ]
                }
            ],
            max_tokens=1024
        )
        return response.choices[0].message.content

    else:  # .txt
        return file.read().decode("utf-8")


def extract_text_from_url(url: str) -> str:
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    except requests.exceptions.RequestException as e:
        return f"ERROR: {e}"
