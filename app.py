from flask import Flask, request, jsonify
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

app = Flask(__name__)

@app.route('/summarize', methods=['POST'])
def summarize_review():
    data = request.get_json()
    youtube_url = data.get("youtube_url")

    prompt = """
You are watching a movie review video. Please analyze only what the YouTuber/reviewer says and summarize the review in the following format:

1. 🎮 Movie Name (if mentioned):
2. 📖 Story Summary (Only what the reviewer explained — no assumptions or generic storylines):
3. ✅ Positives Mentioned (specific points praised):
4. ❌ Negatives Mentioned (specific criticisms):
5. 🧠 Reviewer’s Final Opinion (summary of their recommendation in their words):
6. ⭐ Rating (only if the reviewer clearly gave one):
7. 🎥 Should you watch it or skip it? (according to the reviewer):

Important: Do NOT make up or guess any part. Use only what the reviewer clearly spoke in the video.
"""

    try:
        response = client.models.generate_content(
            model="models/gemini-2.0-flash",
            contents=types.Content(
                parts=[
                    types.Part(file_data=types.FileData(file_uri=youtube_url)),
                    types.Part(text=prompt)
                ]
            )
        )
        return jsonify({"summary": response.text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
