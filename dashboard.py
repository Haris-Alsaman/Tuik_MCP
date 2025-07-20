import os
import json
import jwt
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Flask, request, jsonify, render_template_string

# --- Configuration ---
PRIVATE_KEY_FILE = "private_key.pem"
PUBLIC_KEY_FILE = "public_key.pem"
TOKENS_FILE = "tokens.json"
ISSUER_URL = "https://wagmi.tech/auth"
AUDIENCE = "wagmi-tech-payment-link-mcp-server"

# --- Flask App Initialization ---
app = Flask(__name__)

# --- Key Management ---
def generate_and_save_keys():
    """Generates a new RSA key pair and saves it to PEM files."""
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    public_key = private_key.public_key()

    # Save private key
    with open(PRIVATE_KEY_FILE, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    # Save public key
    with open(PUBLIC_KEY_FILE, "wb") as f:
        f.write(public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    
    return private_key

def load_keys():
    """Loads RSA keys from PEM files, generating them if they don't exist."""
    if not os.path.exists(PRIVATE_KEY_FILE):
        private_key = generate_and_save_keys()
    else:
        with open(PRIVATE_KEY_FILE, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
    
    with open(PUBLIC_KEY_FILE, "rb") as f:
        public_key_pem = f.read()

    return private_key, public_key_pem

# --- Token Management ---
def load_tokens():
    """Loads the list of active tokens from the JSON file."""
    if not os.path.exists(TOKENS_FILE):
        return []
    with open(TOKENS_FILE, "r") as f:
        return json.load(f)

def save_tokens(tokens):
    """Saves the list of active tokens to the JSON file."""
    with open(TOKENS_FILE, "w") as f:
        json.dump(tokens, f, indent=2)

# --- HTML Template ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Token Management Dashboard</title>
    <style>
        body { font-family: sans-serif; margin: 2em; }
        .token { border: 1px solid #ccc; padding: 1em; margin-bottom: 1em; }
        .token p { word-break: break-all; }
    </style>
</head>
<body>
    <h1>Token Management Dashboard</h1>
    
    <h2>Generate New Token</h2>
    <form action="/generate" method="post">
        <label for="subject">User/Client Name (Subject):</label>
        <input type="text" id="subject" name="subject" required>
        <button type="submit">Generate Token</button>
    </form>
    
    <h2>Active Tokens</h2>
    {% for token_info in tokens %}
        <div class="token">
            <p><strong>Subject:</strong> {{ token_info.subject }}</p>
            <p><strong>Token:</strong> <code>{{ token_info.token }}</code></p>
            <form action="/revoke" method="post" style="display:inline;">
                <input type="hidden" name="token_to_revoke" value="{{ token_info.token }}">
                <button type="submit">Revoke</button>
            </form>
        </div>
    {% else %}
        <p>No active tokens.</p>
    {% endfor %}
</body>
</html>
"""

# --- Flask Routes ---
@app.route("/")
def index():
    """Displays the main dashboard with active tokens."""
    tokens = load_tokens()
    return render_template_string(HTML_TEMPLATE, tokens=tokens)

@app.route("/generate", methods=["POST"])
def generate_token_route():
    """Generates a new JWT for a given subject."""
    subject = request.form.get("subject")
    if not subject:
        return "Subject is required", 400
    
    private_key, _ = load_keys()
    
    # Create the token
    token = jwt.encode(
        {
            "iss": ISSUER_URL,
            "sub": subject,
            "aud": AUDIENCE,
            "exp": datetime.utcnow() + timedelta(days=365)
        },
        private_key,
        algorithm="RS256"
    )

    # Save the token
    tokens = load_tokens()
    tokens.append({"subject": subject, "token": token})
    save_tokens(tokens)
    
    return index()

@app.route("/api/generate_token", methods=["POST"])
def api_generate_token():
    """Generates and returns a token as JSON."""
    subject = request.form.get("subject")
    if not subject:
        return jsonify({"error": "Subject is required"}), 400
    
    private_key, _ = load_keys()
    
    token = jwt.encode(
        {
            "iss": ISSUER_URL,
            "sub": subject,
            "aud": AUDIENCE,
            "exp": datetime.utcnow() + timedelta(days=365)
        },
        private_key,
        algorithm="RS256"
    )

    tokens = load_tokens()
    tokens.append({"subject": subject, "token": token})
    save_tokens(tokens)
    
    return jsonify({"token": token})

@app.route("/revoke", methods=["POST"])
def revoke_token_route():
    """Revokes a token by removing it from the active list."""
    token_to_revoke = request.form.get("token_to_revoke")
    tokens = load_tokens()
    tokens = [t for t in tokens if t.get("token") != token_to_revoke]
    save_tokens(tokens)
    
    return index()

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure keys exist before starting
    load_keys()
    app.run(host="0.0.0.0", port=8070, debug=True) 