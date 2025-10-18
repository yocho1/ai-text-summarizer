from flask import Flask, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client with API key from .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to AI Text Summarizer API 🚀"

@app.route("/summarize", methods=["POST"])
def summarize():
    try:
        # Get JSON data from the request
        data = request.get_json()
        text_to_summarize = data.get("text", "")

        if not text_to_summarize.strip():
            return jsonify({"error": "No text provided"}), 400

        # Call OpenAI GPT model
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert summarizer."},
                {"role": "user", "content": f"Summarize this text in a few sentences:\n\n{text_to_summarize}"}
            ],
            max_tokens=150,
            temperature=0.7
        )

        summary = response.choices[0].message.content.strip()
        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
