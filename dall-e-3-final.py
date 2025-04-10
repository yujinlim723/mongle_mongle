from openai import AzureOpenAI
import os
import requests
from PIL import Image
import json
from dotenv import load_dotenv
from datetime import datetime
import time  # 시간 딜레이를 위한 라이브러리 추가

# 환경 변수 로드
load_dotenv()

# Azure OpenAI 클라이언트 설정
client = AzureOpenAI(
    api_version="2024-02-01",  
    api_key=os.environ["AZURE_OPENAI_API_KEY"],  
    azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT']
)

# 프롬프트 설정
prompt = '''
한밤중, 피투성이 손으로 어두운 골목에서 쫓기던 나는 결국 뒤따라오는 사람과 마주쳐 본능적으로 칼을 휘둘러 그를 찔렀다. 그의 죽음을 확인한 후 자리를 떠났지만, 그 순간의 공포와 후회는 평생 나를 따라다닐 것이다
'''

# 최대 재시도 횟수
max_retries = 5
attempt = 0

def save_error_json(error_code, error_response):
    """
    에러 정보를 JSON 파일로 저장하는 함수
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    error_filename = f"error_{error_code}_{timestamp}.json"
    with open(error_filename, 'w', encoding='utf-8') as f:
        json.dump(error_response, f, ensure_ascii=False, indent=2)

# 이미지 생성 요청 및 에러 처리
while attempt < max_retries:
    try:
        result = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1
        )
        break  # 요청 성공 시 루프 종료

    except Exception as e:
        # API에서 반환하는 에러 메시지 파싱
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_response = json.loads(e.response.text)
                error_code = error_response['error']['code']
                error_detail = error_response['error']['inner_error']['content_filter_results']
            except json.JSONDecodeError:
                print("알 수 없는 오류가 발생했습니다.")  # JSON 파싱 실패 시 메시지 출력
                break  # 프로그램 종료

            # jailbreak 에러 처리
            if 'jailbreak' in error_detail and error_detail['jailbreak']['detected']:
                print("DALL-E-3 규제 정책 상 이미지 생성이 어렵습니다.")
                save_error_json(error_code, error_response)
                break  # 프로그램 종료

            # 429 에러 처리: 5초 대기 후 재시도 코드 추가
            if error_code == '429':
                print("현재 가격 정책책 제한으로 인해 요청을 처리할 수 없습니다. 5초 후 다시 시도합니다.")  
                save_error_json(error_code, error_response)  # 에러 정보 저장
                time.sleep(5)  
                attempt += 1  #
                continue  

            # revised_prompt가 있는 경우 재시도
            inner_error = error_response.get('error', {}).get('inner_error', {})
            if 'revised_prompt' in inner_error:
                print("수정된 프롬프트로 재시도합니다.")
                prompt = inner_error['revised_prompt']
                attempt += 1  # 재시도 횟수 증가
                continue
        else:
            print("API 응답이 없습니다.")  # API 응답이 없는 경우
            break

# 재시도 초과 시
if attempt == max_retries:
    print("여러 번 시도했지만 이미지를 생성할 수 없습니다.")

# 요청 성공 시 이미지 처리
if 'result' in locals():  # result 객체가 존재하는 경우에만 처리
    json_response = json.loads(result.model_dump_json())
    image_dir = os.path.join(os.curdir, 'images')

    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    image_path = os.path.join(image_dir, 'generated_image.png')
    image_url = json_response["data"][0]["url"]
    generated_image = requests.get(image_url).content

    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)

    image = Image.open(image_path)
    image.show()