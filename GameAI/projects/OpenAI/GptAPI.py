import openai
import os

api_key = "sk-proj-1WLXANjzW9nOROnZYYxST3BlbkFJf04uXWkXG6tvtpfTDHXG"

openai.api_key = api_key
gpt_engine = "gpt-3.5-turbo-instruct"
gpt_temperature = 0.5
# prompt를 제외하고 생성된 내용이 4096 tokens를 넘어선 안됨
gpt_max_tokens = 3000


def Response(generated):
    return generated.choices[0].text


def TranslateParagraphs(prompt):
    prompt += "translate this in english, Maintain the original honorifics as much as possible when translating."
    return Response(openai.Completion.create(
        engine=gpt_engine,
        prompt=prompt,
        max_tokens=gpt_max_tokens,
        temperature=gpt_temperature,
    ))

def Translate(pargraph):
    translated = TranslateParagraphs(pargraph)
    return translated