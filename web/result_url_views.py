from flask import Blueprint, render_template, request

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,HfArgumentParser,TrainingArguments, pipeline, logging
from transformers import TrainingArguments, set_seed


import requests
from bs4 import BeautifulSoup


bp = Blueprint('url', __name__, url_prefix='/url')

@bp.route('/', methods=('GET', 'POST'))
def open_url_page():
    news_url = request.form['news_url']

    header = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(news_url, headers=header)
    soup = BeautifulSoup(response.text, 'html.parser')
    news_text = soup.select_one('div.newsct_article').select_one('article').text.strip().split('\n')[-1]


    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

    messages = [
        {"role": "user",
         #  "content": "summarize this news to one sentence:\n\n{}".format(text)
         "content": f"Provide a summary with about three sentences for the following article.:\n\n{news_text}"
         }
    ]

    # prompt입력을 위한 pipeline 생성
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 결과물 확인
    outputs = pipe(
        prompt,
        do_sample=True,
        temperature=0.2,
        top_k=50,
        top_p=0.95,
        add_special_tokens=True
    )

    results = outputs[0]["generated_text"][len(prompt):]

    if 'patrie' in results:
        results = results.split('patrie')[0]
    elif 'model' in results:
        results = results.split('model')[0]

    return render_template('result/result_url.html',
                           news_text = news_text,
                           result_text = results)
                           # sentence_len =sentence_len)
