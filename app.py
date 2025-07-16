from flask import Flask, request, jsonify
from KPG import run_llm
from flask_cors import CORS
from datetime import datetime
from pathlib import Path
import csv


app = Flask(__name__)
CORS(app)

user_sessions = {}

# Load the API key from environment variables or set it directly
API_KEY = "K1ovn39nsfs49mlsg"

def get_user_id():
    user_ip = request.remote_addr
    user_agent = request.headers.get('User-Agent', 'unknown')
    return f"{user_ip}_{hash(user_agent)}"

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json

    # Validate request structure
    if not data or "api_key" not in data or "query" not in data:
        return jsonify({"error": "Invalid request. 'api_key' and 'query' are required."}), 400

    # Validate API key
    if data["api_key"] != API_KEY:
        return jsonify({"error": "Unauthorized. Invalid API key."}), 403

    # Get the query
    query = data["query"]
    user_id = get_user_id()

    try:
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        history = _get_user_history(log_folder_path=log_path, user_id=user_id)
        response = run_llm(query=query, chat_history=history, user_id=user_id)

        return jsonify({"answer": response}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def _get_user_history(log_folder_path, user_id):
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = Path(f"{log_folder_path}/{date_str}.csv")

    history = []

    if not filename.exists():
        return "unavailable"

    with open(filename, mode="r", newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["user_id"] == user_id:
                history.append(("user", row['query']))
                history.append(("ai", row['response']))

    return history if history else []


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
