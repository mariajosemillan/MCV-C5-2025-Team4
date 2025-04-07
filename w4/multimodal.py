import torch
import json
import pandas as pd
import os
print(torch.__version__)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "/ghome/c5mcv04/MCV-C5-2025-Team4/dataset/foodDataset/Food_Images/-bloody-mary-tomato-toast-with-celery-and-horseradish-56389813.jpg",
#             },
#             {"type": "text", "text": ""},
#         ],
#     }
# ]

with open("/ghome/c5mcv04/messages_QWEN3.json", 'r') as f:
    messages = json.load(f)
df = pd.DataFrame(columns=["Image_Name","PRED_Title"])
for i in range(len(messages)):
    for content_item in messages[i]['content']:
        if content_item['type'] == 'image':
            image_path = content_item['image']
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            print(image_name)

    text = processor.apply_chat_template(
        [messages[i]], tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info([messages[i]])
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    title = output_text[0]
    cleaned_title = title.replace("*", "").replace('"', "").replace("'", "")
    print(type(output_text[0]))
    print(cleaned_title)
    df.loc[len(df)] = [image_name, cleaned_title]

df.to_csv("output_file_QWEN3.csv", index=False)
