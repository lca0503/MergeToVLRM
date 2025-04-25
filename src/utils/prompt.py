from .const import VLMScoringPromptEnum


def format_text_only_prompts(query, responses, tokenizer):
    messages = [
        [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response}
        ] for response in responses
    ]
    prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return prompts


def format_vlm_only_prompts(query, responses, processor, num_images=1):
    prompts = []
    for response in responses:
        system_prompt = VLMScoringPromptEnum.system_prompt.strip()
        user_prompt = VLMScoringPromptEnum.user_prompt.format(
            INSTRUCTION="<|image|>" * num_images + query,
            ANSWER=response,
        )

        message = [
            {"role": "system",
             "content": system_prompt,
            },
            {"role": "user",
             "content": [
                 {"type": "text", "text": user_prompt}
             ]
            }
        ]
        prompt = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
        
    return prompts    


def format_no_image_prompts(query, responses, processor):
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ]
            }
        ] 
        for response in responses
    ]
    prompts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return prompts


def format_prompts(query, responses, processor, num_images=1):
    messages = [
        [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in range(num_images)],
                    {"type": "text", "text": query}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response}
                ]
            }
        ] 
        for response in responses
    ]
    prompts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return prompts


