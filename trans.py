import numpy as np
from transformers import AutoProcessor, SeamlessM4TModel

processor = AutoProcessor.from_pretrained("facebook/hf-seamless-m4t-medium")
model = SeamlessM4TModel.from_pretrained("facebook/hf-seamless-m4t-medium")

def translate_text(contents):

    # Process the text
    text_inputs = processor(text = contents, src_lang="eng", return_tensors="pt")

    # Translate the text
    output_tokens = model.generate(**text_inputs, tgt_lang="arb", generate_speech=False)
    
    # Decode the translated text
    translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

    return translated_text_from_text

# Usage example:
translate_text( "The report prepared by the Tibetan Institute of Hydrology at the Chinese Academy of Sciences suggests that temperatures and humidity levels continue to rise throughout this century.")
