import argparse
import glob
import json
import os
from pathlib import Path
from transformers import  AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--repo_id", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--model_name", type=str,default="Qwen/Qwen3-Coder-30B-A3B-Instruct")
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = args.input_dir
    output_path = args.output_path
    repo_id = args.repo_id
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    total_code_path_lst = glob.glob(input_dir+"/**/*", recursive=True)
    
    # 결과를 저장할 리스트
    dataset = []
    check_token_size_list = []
    for code_path in total_code_path_lst:
        # 디렉토리는 건너뛰기
        if os.path.isdir(code_path):
            continue
            
        try:
            with open(code_path, "r") as f:
                code = f.read()
        except:        
            with open(code_path, "r", encoding="cp949") as f:
                code = f.read()
        
        # 파일 경로를 상대 경로로 변환
        file_path = os.path.relpath(code_path, input_dir)
        # 파일 크기 계산
        file_size = os.path.getsize(code_path)
        
        # 스키마에 맞는 데이터 구조 생성
        data_entry = {
            "repo_id": repo_id,
            "file_path": file_path,
            "content": code,
            "size": file_size,
            "token_size": len(tokenizer.encode(code))
        }

        dataset.append(data_entry)
        check_token_size_list.append([len(tokenizer.encode(code)),file_path])
    # JSON 파일로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 총 {len(dataset)}개 파일을 {output_path}에 저장했습니다.")
    max_token_entry = max(check_token_size_list, key=lambda x: x[0])
    min_token_entry = min(check_token_size_list, key=lambda x: x[0])
    avg_token_size = sum([entry[0] for entry in check_token_size_list]) / len(check_token_size_list)
    
    print(f"토큰 크기 최대값: {max_token_entry[0]} (파일: {max_token_entry[1]})")
    print(f"토큰 크기 최소값: {min_token_entry[0]} (파일: {min_token_entry[1]})")
    print(f"토큰 크기 평균값: {avg_token_size:.2f}")



if __name__ == "__main__":
    main()