from flask import Blueprint, render_template, request

import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging
from transformers import TrainingArguments, set_seed

import re

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

set_seed(42)

# 불러올 모델 경로 - base model
new_model = 'jental/gemma_model10000'

# 허깅페이스 토큰
HUGGINGFACE_AUTH_TOKEN = ''

# bnb 환경 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)  # 4bit quantization

# 토크나이저 불러오기
# tokenizer를 먼저 불러와야 한다
tokenizer = AutoTokenizer.from_pretrained(new_model, token=HUGGINGFACE_AUTH_TOKEN, padding_side="right")

# 모델 불러오기
model = AutoModelForCausalLM.from_pretrained(
    new_model,
    token=HUGGINGFACE_AUTH_TOKEN,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = True

model1_error_delete = ['user',
                       r'\bmunicipi\b'  # bmunicipi로 시작하는 단어들
                       ]

model1_error_split = [
    'model',
    'modelrehension',
    'patrie:',
    'centrifuga',
    'modelrehension · 100자 제출.'
]


bp = Blueprint('result', __name__, url_prefix='/result')


@bp.route('/text', methods=('GET', 'POST'))
def open_result_page():
    origin_text = request.form['origin_text']

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

    # if request.form['sentence'] == 'sentenceType1':
    #     sentence_len = '1줄요약'
    #
    #     messages = [
    #         {"role": "user",
    #          #  "content": "summarize this news to one sentence:\n\n{}".format(text)
    #          "content": f"Provide a summary with about one sentence of one hundred characters for the following article.:\n\n{origin_text}"
    #          }
    #     ]
    # elif request.form['sentence'] == 'sentenceType3':
    #     sentence_len = '3줄요약'
    #     messages = [
    #         {"role": "user",
    #          #  "content": "summarize this news to one sentence:\n\n{}".format(text)
    #          "content": f"Provide a summary with about three sentences for the following article.:\n\n{origin_text}"
    #          }
    #     ]

    messages = [
        {"role": "user",
         #  "content": "summarize this news to one sentence:\n\n{}".format(text)
         "content": f"Provide a summary with about three sentences for the following article.:\n\n{origin_text}"
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

    ### 모델에서 발생하는 오류들을 수정하는 항목



    # 특정 단어가 등장하면 오류가 발생
    for word in model1_error_split:
        if word in results:
            results = results.split(word)[0]

    # 오류가 발생하는 단어
    for word in model1_error_delete:
        if word in results:
            results = re.sub(word, '', results)





    return render_template('result/result.html',
                           origin_text=origin_text,
                           result_text=results)
    # sentence_len =sentence_len)


#### url 입력시 -> 뉴스 기사 크롤링 하여 바로 요약

@bp.route('/url', methods=('GET', 'POST'))
def open_url_page():
    news_url = request.form['news_url']

    # 스포츠 / 엔터는 방식이 다르다
    # bs4로 크롤링이 불가능 -> selenium 사용
    # font 문제로 잘 안된다
    if 'sports' in news_url or 'entertain' in news_url:

        chrome_options = Options()
        chrome_options.add_argument('headless')

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(news_url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        # news_text = driver.find_element(By.CLASS_NAME, '_article_content').text.strip().replace('\n', '')
        news_text = soup.select_one('article').select_one('div._article_content').text
        
    # 나머지 뉴스는 속도를 위해서 bs4 사용
    else:
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

    ### 모델에서 발생하는 오류들을 수정하는 항목

    # 특정 단어가 등장하면 오류가 발생
    for word in model1_error_split:
        if word in results:
            results = results.split(word)[0]

    # 오류가 발생하는 단어
    for word in model1_error_delete:
        if word in results:
            results = re.sub(word, '', results)


    return render_template('result/result_url.html',
                           news_text=news_text,
                           result_text=results)
    # sentence_len =sentence_len)
