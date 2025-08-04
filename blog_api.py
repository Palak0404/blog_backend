from flask import Flask, request, jsonify
import google.generativeai as genai
from openai import AzureOpenAI
import os

app = Flask(__name__)

# === Gemini API Key Configuration ===
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# === Azure GPT-4o Configuration ===
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("ENDPOINT_URL")
AZURE_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME")
AZURE_API_VERSION = "2025-01-01-preview"

client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

@app.route('/generate_blog', methods=['POST'])
def generate_blog():
    data = request.get_json(force=True)
    data.pop("prompt", None)  # Remove custom prompt if passed accidentally

    model_choice = data.get("model", "gemini").lower()
    topic = data.get("topic", "").strip()

    print(f">>> Request received — Model: {model_choice}, Topic: {topic}")

    if not topic:
        return jsonify({"error": "Missing or empty topic"}), 400

    # Unified, structured yet human-toned prompt
    prompt = f"""
Write at a 5th grade level. Use clear, simple language and natural phrasing — like how people talk in everyday conversation. Keep the words easy, the tone chill, and the vibe friendly.

## Writing Style:
- Make it personal, casual, and engaging — like a real person talking.
- Use paragraphs, natural pacing, and real-world examples.
- Use simple language and avoid technical jargon.
- Add human touches like relateable and conversational phrases.

## Structure :
1. A short, relatable 2-3 sentence introduction.
2. A catchy H1 title using #
3. A Markdown-formatted Table of Contents with 4-6 H2 sections (##), numbered.
4. Each H2 section should contain:
   - Two subpoints using ### with ~150-200 words total.
5. A ## Frequently Asked Questions section:
   - 4 common questions with short, helpful 2-3 sentence answers.
6. A ## Conclusion (2-3 sentences to wrap up).

## Format:
Use proper Markdown. Output only the blog — no extra explanations.

## Topic:
{topic}
"""

    try:
        if model_choice == "gemini":
            print("Using Gemini model")
            model = genai.GenerativeModel("models/gemini-2.0-flash")
            stream = model.generate_content(
                prompt,
                stream=True,
                generation_config={
                    "temperature": 0.9,
                    "top_k": 50,
                    "top_p": 0.95,
                    "max_output_tokens": 4096
                }
            )

            full_blog = ""
            for chunk in stream:
                if chunk.text:
                    full_blog += chunk.text

        elif model_choice == "gpt-4o-azure":
            print("Using Azure GPT-4o model")

            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You're a comman human . You follow a clear blog structure with TOC, FAQs, and Conclusion. "
                            "Your tone is friendly, casual, and conversational — written for 5th-grade readers. "
                            "Avoid robotic, generic, or overly formal writing."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.9,
                top_p=0.95,
                max_tokens=2048
            )

            full_blog = response.choices[0].message.content

        else:
            return jsonify({"error": "Invalid model specified. Use 'gemini' or 'gpt-4o-azure'."}), 400

        return jsonify({"blog": full_blog})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
