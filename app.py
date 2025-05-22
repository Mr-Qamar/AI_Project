import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from flask import Flask, request, jsonify
import flask_cors
from flask_cors import CORS

# Flask app ko Initialize krna hy
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True, 
     allow_headers="*", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"])

# gpt2 
model_name = "gpt2"

# preprocessing krni input ki
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# cpu pr training set krni agr gpu nahi to
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    
    # user messege model ko dena agr na day to default value deni hy
    prompt = data.get('message', "how are you?")
    max_length = 50 
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.get('attention_mask', None),
            max_length=max_length
        )
    
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    response = text.split("\n\n")
    
    return jsonify({
        "message": response[1]
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
    

