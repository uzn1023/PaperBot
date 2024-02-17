import openai
import os
import secret_keys
import streamlit as st

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) 

# openai.api_key = secret_keys.openai_api_key
openai.api_key = st.secrets.OpenAIAPI.openai_api_key

def get_completion(prompt, model="gpt-3.5-turbo"): 
    messages = [{"role": "user", "content": prompt}]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message.content

def get_translation(prompt):
    content = f"""
    {prompt}
    """

    prompt = f"""
    # あなたは与えられた文を日本語に翻訳してください\
    # 難しい専門用語は英単語のままでも構いません\
    # （三重のバッククォートで区切られている部分）を日本語に翻訳してください。\
    # タイトルや著者名など文章以外の部分は改行してください\
    # 適切に改行を行ってください\

    ```{content}```
    """

    response = get_completion(prompt)
    return response