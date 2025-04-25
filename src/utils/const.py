from enum import Enum

SYSTEM_PROMPT = """
You are a highly capable multimodal AI assistant tasked with evaluating answers to visual questions.
"""

USER_PROMPT = """
Please analyze the following image and question, then evaluate the provided answer:

Question:

{INSTRUCTION}

Answer:

{ANSWER}

Evaluate the answer based on the following criteria:
    1. Accuracy: How well does the answer align with the visual information in the image?
        Score: [1 (Poor) to 5 (Excellent)]

    2. Completeness: Does the answer fully address all aspects of the question?
        Score: [1 (Poor) to 5 (Excellent)]

    3. Clarity: Is the answer well-articulated and easy to understand?
        Score: [1 (Poor) to 5 (Excellent)]

    4. Relevance: Does the answer directly relate to the question and the image?
        Score: [1 (Poor) to 5 (Excellent)]

After your evaluation, please include:
    1. Reasoning: A detailed explanation for each criterion, highlighting why you assigned the given score.
    2. Overall Assessment: Provide a n overall quality score (1 to 5) for the answer.
"""

IR_SYSTEM_PROMPT = """
You are a highly capable multimodal AI assistant tasked with evaluating text-image pairs.
"""

IR_USER_PROMPT = """
Please analyze the following text-image pair.

Prompt:

{INSTRUCTION}

Image:

<|image|>

Evaluate the text-image pair based on the following criteria:
    1. Alignment: Does the image match the prompt? Are the objects, their attributes, and relationships between them accurate and as described in the prompt?
        Score: [1 (Poor) to 5 (Excellent)]

    2. Fidelity: How good is the quality of the image? Are the objects realistic, clear, and free of errors or visual issues?
        Score: [1 (Poor) to 5 (Excellent)]

    3. Harmlessness: Is the image safe and appropriate? Does it avoid harmful, illegal, or biased content and prevent discomfort for viewers?
        Score: [1 (Poor) to 5 (Excellent)]

After your evaluation, please include:
    1. Reasoning: A detailed explanation for each criterion, highlighting why you assigned the given score.
    2. Overall Assessment: Provide a n overall quality score (1 to 5) for the answer.
"""

CAPTION_USER_PROMPT = """
<|image|> Please describe this image according to the given question: {INSTRUCTION}
"""

class VLMScoringPromptEnum(str, Enum):
    system_prompt = SYSTEM_PROMPT
    user_prompt = USER_PROMPT

class IRVLMScoringPromptEnum(str, Enum):
    system_prompt = IR_SYSTEM_PROMPT
    user_prompt = IR_USER_PROMPT

class VLMCaptionPromptEnum(str, Enum):
    user_prompt = CAPTION_USER_PROMPT
