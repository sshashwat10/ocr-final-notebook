

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig
from transformers import BitsAndBytesConfig
import accelerate
# quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)
bnb_config = BitsAndBytesConfig(
    llm_int8_enable_fp32_cpu_offload=True,
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
# control model memory allocation between devices for low GPU resource (0,cpu)
device_map = {
    "transformer.word_embeddings": 0,
    "transformer.word_embeddings_layernorm": 0,
    "lm_head": 0,
    "transformer.h": 0,
    "transformer.ln_f": 0,
    "model.embed_tokens": 0,
    "model.layers":0,
    "model.norm":0
}
device = "cuda" if torch.cuda.is_available() else "cpu"

# model use for inference
model_id="mychen76/mistral7b_ocr_to_json_v1"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    device_map=device_map)
# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)



from paddleocr import PaddleOCR

# Initialize PaddleOCR
paddleocr = PaddleOCR(lang="en", ocr_version="PP-OCRv4", use_gpu=True, show_log=False)

def paddle_scan(paddleocr, img_path_or_nparray):
    result = paddleocr.ocr(img_path_or_nparray, cls=True)
    result = result[0]  # Assuming only one image is processed
    boxes = [line[0] for line in result]       # Bounding boxes
    txts = [line[1][0] for line in result]     # Raw text
    scores = [line[1][1] for line in result]   # Scores
    return txts, result

# Function to perform OCR and extract JSON
def process_invoice(image_path):
    try:
        # Perform OCR
        receipt_texts, receipt_boxes = paddle_scan(paddleocr, image_path)
        print(receipt_texts)
        print(receipt_boxes)
        # Prepare prompt for Hugging Face model
        prompt = f"""### Instruction:
        You are POS receipt data expert, parse, detect, recognize and convert following receipt OCR image result into structure receipt data object.
        Don't make up value not in the Input. Output must be a well-formed JSON object.```json


        ### Input:
        {receipt_boxes}

        ### Output:
        """

        # Generate JSON output using Hugging Face model
        with torch.inference_mode():
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(**inputs, max_new_tokens=512)
            result_text = tokenizer.batch_decode(outputs)[0]
            print(result_text)

        return {
            # "ocr_text": receipt_texts,
            "json_output": result_text
        }

    except Exception as e:
        return {
            "error": str(e)
        }

from google.colab import drive
drive.mount('/content/drive')



# Example usage
import urllib.request
import numpy as np
from PIL import Image

if __name__ == "__main__":
    image_path = '/content/drive/MyDrive/ocr_prac.jpeg'
    image_url='https://groups.google.com/group/jzebra-users/attach/d16dbba8a612edfa/Bill%20Image_Receipt.png?part=0.1'
    local_image_id='bill_image_receipt.png'
    urllib.request.urlretrieve(image_url,local_image_id)
    receipt_image = Image.open("/content/drive/MyDrive/img_2.png")
    receipt_image_array = np.array(receipt_image.convert('RGB'))
    display(receipt_image.resize((300,400)))  # Display the image
    result = process_invoice(receipt_image_array)
    prompt_str = str(result).replace("\n", "")
    output_index = prompt_str.find("### Output:")

    # Extract everything after "### Output:"
    if output_index != -1:
        output_content = prompt_str[output_index + len("### Output:"):]
        output_content=output_content.replace("\n", "")
        print(output_content)
    else:
        print("### Output: not found in the prompt.")
    # ocr=extract_text_before_output(result)
    print(type(result))


from flask import Flask,jsonify,request
from pyngrok import ngrok
from werkzeug.utils import secure_filename

port_no = 5000

import os
app = Flask(__name__)
ngrok.set_auth_token("2elGda95W53NMZfKTs5E5JqySJu_43ZrZJQN7xQbUf6aLM7Lj")
public_url =  ngrok.connect(port_no).public_url
import numpy as np
import cv2


from google.colab import drive
drive.mount('/content/drive')


# Define the upload folder and ensure it exists
UPLOAD_FOLDER = '/content/drive/MyDrive/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def home():
    return f"Running Flask on Google Colab!"

print(f"To acces the Gloable link please click {public_url}")

@app.route("/upload", methods=["GET"])
def upload_file():
    image_path = '/content/drive/MyDrive/ocr_prac.jpeg'
    image_url='https://groups.google.com/group/jzebra-users/attach/d16dbba8a612edfa/Bill%20Image_Receipt.png?part=0.1'
    local_image_id='bill_image_receipt.png'
    urllib.request.urlretrieve(image_url,local_image_id)
    receipt_image = Image.open(local_image_id)
    receipt_image_array = np.array(receipt_image.convert('RGB'))
    result = process_invoice(receipt_image_array)
    print(result)
    return jsonify(result)

UPLOAD_FOLDER = '/content/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload_ocr', methods=['POST'])
def upload_and_ocr():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Perform OCR on the uploaded file
        receipt_image = Image.open(file_path)
        receipt_image_array = np.array(receipt_image.convert('RGB'))
        ocr_result = process_invoice(receipt_image_array)

        #convert dict to string
        prompt_str = str(ocr_result)
        output_index = prompt_str.find("### Output:")

        # Extract everything after "### Output:"
        if output_index != -1:
            output_content = prompt_str[output_index + len("### Output:"):]

            #print(output_content)
        else:
            print("### Output: not found in the prompt.")
        return jsonify({"message": "File successfully uploaded", "filename": filename, "ocr_result": output_content}), 200

app.run(port=port_no)

