import re

import torch


def find_numbers(text):
    numbers = re.compile(
        r'-?[\d,]*\.?\d+',
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(text)
    
    return numbers


def find_number(text, answer_delimiter):
    if answer_delimiter in text:
        answer = text.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]
        
    numbers = find_numbers(text)
    if numbers:
        return numbers[-1]
    return None


def remove_comma(x):
    return x.replace(',', '')


def text_only_scoring(prompts, model, tokenizer):
    scores = []
    for prompt in prompts:
        inputs = tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            scores.append(
                round(float(outputs.logits.cpu().item()), 5)
            )
            
    return scores


def vlm_only_scoring(prompts, images, model, sampling_params):
    vlm_outputs = []
    scores = []
    inputs = []
    for prompt in prompts:
        inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": images},
        })

    outputs = model.generate(inputs, sampling_params)
    text_outputs = [out.outputs[0].text for out in outputs]
    for text_output in text_outputs:
        score = find_number(text_output, answer_delimiter="Overall")
        if not score:
            score = "0.0"
        score = remove_comma(score)
        scores.append(round(float(score), 5))        
    return vlm_outputs, scores


def no_image_scoring(prompts, model, processor):
    scores = []
    for prompt in prompts:
        inputs = processor(
            text=prompt,
            images=None,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
            
        with torch.no_grad():
            outputs = model(**inputs)
            scores.append(
                round(float(outputs.logits.cpu().item()), 5)
            )

    return scores


def scoring(prompts, images, model, processor):
    scores = []
    for prompt in prompts:
        inputs = processor(
            text=prompt,
            images=images,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
            
        with torch.no_grad():
            outputs = model(**inputs)
            scores.append(
                round(float(outputs.logits.cpu().item()), 5)
            )

    return scores

