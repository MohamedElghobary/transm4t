from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import numpy as np
from transformers import AutoProcessor, SeamlessM4TModel
from langdetect import detect


app = Flask(__name__)
run_with_ngrok(app)


processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")


def detect_language(input_text):
    try:
        language_code = detect(input_text)
        return language_code
    except Exception as e:
        # Handle exceptions (e.g., if language detection fails)
        return None

def translate_text(contents, src_lang=None, tgt_lang="arb"):
    if src_lang is None:
        detected_language = detect_language(contents)
        if detected_language is not None:
            src_lang = detected_language
        else:
            raise ValueError("Failed to detect the input language.")
    
    # Process the text
    text_inputs = processor(text=contents, src_lang=src_lang, return_tensors="pt")

    # Translate the text
    output_tokens = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)

    # Decode the translated text
    translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

    return translated_text

# Usage examples:

# input_text = "The report prepared by the Tibetan Institute of Hydrology at the Chinese Academy of Sciences suggests that temperatures and humidity levels continue to rise throughout this century."
# translated_text = translate_text(input_text, src_lang=None, tgt_lang="arb")
# print("Translated text (from detected language to Arabic):", translated_text)

# input_text_arabic = "تقرير أعده معهد التبت للجليد في الأكاديمية الصينية للعلوم يشير إلى أن درجات الحرارة ومستويات الرطوبة تستمر في الارتفاع طوال هذا القرن."
# translated_text_english = translate_text(input_text_arabic, src_lang=None, tgt_lang="eng")
# print("Translated text (from detected language to English):", translated_text_english)


@app.route('/translate', methods=['POST'])
def translate():
    
    data = request.get_json()
    input_text = data['text']
    try:
        if input_text:
            input_text = data['text']
            target_language = data.get('target_language', 'arb')
            translated_text = translate_text(input_text, tgt_lang=target_language)
            return jsonify({'translated_text': translated_text})
        else :
            return jsonify({"message":"something went wrong"})
    except Exception as e:
        return jsonify({'error': str(e)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)


    