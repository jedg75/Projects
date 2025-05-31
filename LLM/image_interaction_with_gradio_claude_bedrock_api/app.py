import gradio as gr
import requests
import json
import base64
import io

# Dummy API configuration (replace with your actual API details)
API_KEY = "your_api_key_here"
MODEL_ID = "example-model-id"
API_URL = "https://api.example.com/model/invoke"

# Headers for API request
headers = {
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

# Function to convert image to base64 string
def encode_image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Main function to process image and text
def process_image_and_text(image, text):
    content = []

    # Add image to content if provided
    if image:
        encoded_image = encode_image_to_base64(image)
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": encoded_image,
            }
        })

    # Add text to content if provided
    if text:
        content.append({"type": "text", "text": text})

    # Prepare the request payload
    payload = {
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ]
    }

    # Send request to API
    response = requests.post(API_URL, headers=headers, json=payload)
    response_data = response.json()

    # Extract and return the response text
    return response_data['content'][0]['text']

# Create Gradio interface
def create_interface():
    # Define input components
    image_input = gr.Image(label="Upload an Image (Optional)", type="pil")
    text_input = gr.Textbox(label="Enter your question", lines=2)

    # Define output component
    output = gr.Textbox(label="Response", lines=5, max_lines=20)

    # Create the interface
    interface = gr.Interface(
        fn=process_image_and_text,
        inputs=[image_input, text_input],
        outputs=output,
        title="Image and Text Question Answering",
        description="Upload an image and ask a question, or just ask a question without an image."
    )

    # Launch the interface
    interface.launch()

# Run the application
if __name__ == "__main__":
    create_interface()
