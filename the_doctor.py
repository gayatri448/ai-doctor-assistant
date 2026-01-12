# ------------------- Step 1: Setup GROQ API Key -------------------
import os
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# ------------------- Step 2: Encode Image -------------------
import base64

def encode_image(image_path: str) -> str:
    """
    Encodes an image file to base64 string for sending to Groq multimodal LLM.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        str: Base64-encoded image string.
    """
    try:
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded
    except Exception as e:
        print(f"Error encoding image: {e}")
        return ""


# ------------------- Step 3: Analyze Image with Query -------------------
from groq import Groq

def analyze_image_with_query(query: str, encoded_image: str, model: str = "meta-llama/llama-4-scout-17b-16e-instruct"):
    """
    Sends a text + image query to a multimodal LLM via Groq API and returns response.
    
    Args:
        query (str): Text query to ask the LLM.
        encoded_image (str): Base64-encoded image string.
        model (str): Groq model name.
    
    Returns:
        str: LLM response text.
    """
    try:
        client = Groq(api_key=GROQ_API_KEY)  # pass API key here if needed
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ]
        
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return "Could not analyze image."
        

