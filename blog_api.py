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
    topic = data.get("topic", "").strip()
    model_choice = data.get("model", "gemini").lower()  # default: gemini

    print(f">>> Request received — Model: {model_choice}, Topic: {topic}")

    if not topic:
        return jsonify({"error": "Missing or empty topic"}), 400

    prompt = f"""
    You are an expert SEO blog writer. Follow my instructions **exactly**.

    ## Instructions:
    1. Start with a 2–3 sentence introduction to the blog topic.
    2. Use one main H1 title using `#`.
    3. Create a Table of Contents using 4–6 H2 sections (numbered, Markdown format).
    4. Then expand each of those `##` H2s with:
    - 2 `###` H3s inside each
    - ~150–200 words per H2 section total
    5. Add a `## Frequently Asked Questions` section at the end:
    - Include **4 related questions**, with **short 2–3 sentence answers**.
    6. Add a `## Conclusion` (2–3 sentences to wrap up).

    ## Formatting Rules:
    - Use Markdown formatting
    - DO NOT skip any section
    - DO NOT write explanation or extra text — only output the blog

    ## Topic:  
    **{topic}**
    """

    try:
        if model_choice == "gemini":
            print("⚡ Using Gemini model")
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
            print("⚡ Using Azure GPT-4o model")

            response = client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": "You are an expert SEO blog writer."},
                    {"role": "user", "content": prompt}
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
        print("❌ Error:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
