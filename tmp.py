from flask import Flask, render_template, jsonify, redirect, url_for, session, request
from flask_sqlalchemy import SQLAlchemy
import os
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
import threading
from flask_socketio import SocketIO, emit
from datetime import datetime
#from boardUpload import add_dream_to_db, Dream, db  # boardUpload.py에서 가져오기
from openai import AzureOpenAI
import json
import re
import time
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 가져오기
AZURE_OPENAI_ENDPOINT = 'https://team2-openai-eastus2.openai.azure.com/'
AZURE_OPENAI_API_KEY = 'WVyPxTvMy6Z3uAcZQvBi41lmMfpgv75zkiKO8egAhqFonAf0pjSbJQQJ99ALACHYHv6XJ3w3AAABACOG0SVa'



if not AZURE_OPENAI_ENDPOINT or not AZURE_OPENAI_API_KEY:
    raise ValueError("환경 변수가 제대로 설정되지 않았습니다. .env 파일을 확인하세요.")

# AzureOpenAI 클라이언트 초기화
client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version="2024-05-01-preview"
)
def gpt_response(user_input): 
    try:
        logger.info("Generating GPT response")
        assistant = client.beta.assistants.create(            
            model="team2-gpt-mini",  
            instructions="""
            ## Role
            You are an expert in dream interpretation who provides symbolic meanings and concise interpretations of dreams based on the user's input. 
            Respond in the following JSON format.
            ## JSON Format
            {
            "interpretation": ""
            ---
            ## Instructions
            1. The **interpretation** must always be written in **Korean**.
            2. Use polite and formal language (**존댓말**) for the interpretation.
            3. The interpretation should be concise and limited to approximately **2 sentences**.
            4. Use RAG (retrieval-augmented generation) to find symbolic meanings of dreams and incorporate them into the interpretation naturally.
            5. If no relevant data is found via RAG, create an original interpretation based on the user's dream input while strictly adhering to all restrictions.
            ---
            ## Restrictions
            1. Avoid mentioning "dream" as a theme or referring to waking up from a dream.
            ---
            ## Anonymity Guidelines
            1. Do not mention real-world locations (cities, countries, specific place names) or famous people (real individuals or well-known characters).
            2. Replace user-provided names or locations with fictional names or abstract descriptions.
            ---
            ### Important
            Adhere strictly to all instructions and restrictions outlined above.""",
            tools=[{"type":"file_search","file_search":{"ranking_options":{"ranker":"default_2024_08_21","score_threshold":0}}}],
            tool_resources={"file_search":{"vector_store_ids":["vs_pbD6MOjCO9mhML4X474CoAHQ"]}},
            temperature=1,
            top_p=1
          
        )
        thread = client.beta.threads.create()
    
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input
        )
        
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant.id
        )
        
        while run.status in ['queued', 'in_progress', 'cancelling']:
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id
            )
      
        if run.status == 'completed':
            logger.info("GPT response generated successfully")
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            
            for message in reversed(messages.data):
                if message.role == 'assistant':
                    response = message.content[0].text.value
                    
                    ## 뒤에 참고자료가 나와서 정규표현식 사용
                    cleaned_response = re.sub(r'【.*?†source】', '', response)
                    response_json = json.loads(cleaned_response)
                    return response_json
                  
        else:
            logger.error(f"GPT response generation failed with status {run.status}")
            return f"Error: Run failed with status {run.status}"
    except Exception as e:
        logger.error(f"Error during GPT response generation: {e}")
        return f"Error: {str(e)}"
   
def generate_story():
    user_input = "말도 안되는 크기의 숲이 있었고 거기서 가장 큰 나무를 발견했어"
    if not user_input:
        return jsonify({"error": "No input provided"}), 400
    response = gpt_response(user_input)
    print(response)

generate_story()
