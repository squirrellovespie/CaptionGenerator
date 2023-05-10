
import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_step(image_path, num_captions=3, max_length=30, num_beams=5):
    images = []
    for i in range(num_captions):
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        images.append(i_image)
    
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(
        pixel_values,
        do_sample=True,
        num_return_sequences=num_captions,
        max_length=max_length,
        num_beams=num_beams,
        temperature=1.0,
        top_p=0.9,
        top_k=0,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.cls_token_id
    )
    captions = []
    for i in range(num_captions):
        preds = tokenizer.decode(output_ids[i], skip_special_tokens=True)
        captions.append(preds.strip())
    return captions

def generation(image_path): 
    num_captions = 3
    captions = predict_step(image_path, num_captions=num_captions)
    return captions

# UI
st.title("Image Caption Generator")

def validate_file(file):
    if file is None:
        st.warning("Please upload an image.")
        return False
    elif not file.type.startswith("image"):
        st.warning("Please upload an image file.")
        return False
    else:
        return True
    
file = st.file_uploader("Upload file", type=["jpg", "jpeg", "png"])  
if validate_file(file):
    file_name = file.name
    image = file.read()
    captions = generation(file_name) 
    st.image(image, use_column_width=True)
    st.title("-:Captions:-")
    for caption in captions:   
        st.write(caption)