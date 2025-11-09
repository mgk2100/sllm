"""
GitHub 2025 데이터셋 필터링 스크립트
특정 언어의 코드 파일만 수집하여 JSON으로 저장합니다.

사용 예시:
python filtering_github2025.py --languages python c cpp --sample_size 500 --output_path data/output/github2025.json
"""

from datasets import load_dataset
from argparse import ArgumentParser
import json
import os
from typing import List

# 허용된 언어와 해당 파일 확장자 매핑
ALLOWED_LANGUAGES = {
    'python': ['.py'],
    'javascript': ['.js', '.jsx'],
    'typescript': ['.ts', '.tsx'],
    'java': ['.java'],
    'c': ['.c', '.h'],
    'cpp': ['.cpp', '.hpp', '.cc', '.hh', '.cxx', '.hxx'],
    'go': ['.go'],
    'rust': ['.rs'],
    'ruby': ['.rb'],
    'php': ['.php'],
    'swift': ['.swift'],
    'kotlin': ['.kt', '.kts'],
    'scala': ['.scala'],
    'r': ['.r', '.R'],
    'shell': ['.sh', '.bash'],
}


def get_language_from_filepath(file_path: str) -> str:
    """
    파일 경로에서 확장자를 추출하여 언어를 추론합니다.
    
    Args:
        file_path: 파일 경로
        
    Returns:
        언어 이름 또는 None (매칭되지 않을 경우)
    """
    # 파일 확장자 추출 (소문자로 변환)
    ext = os.path.splitext(file_path.lower())[1]
    
    # 확장자로 언어 찾기
    for language, extensions in ALLOWED_LANGUAGES.items():
        if ext in extensions:
            return language
    
    return None


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = ArgumentParser(description="GitHub 2025 데이터셋에서 특정 언어 코드 수집")
    
    parser.add_argument(
        "--languages",
        type=str,
        nargs='+',  # 리스트로 여러 언어 받기
        required=True,
        help=f"수집할 언어 목록 (허용: {', '.join(ALLOWED_LANGUAGES.keys())})"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/output/github2025.json",
        help="출력 JSON 파일 경로"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100,
        help="수집할 샘플 개수"
    )
    
    return parser.parse_args()


def validate_languages(languages: List[str]):
    """
    입력된 언어가 허용된 목록에 있는지 검증합니다.
    
    Args:
        languages: 검증할 언어 리스트
        
    Raises:
        AssertionError: 허용되지 않은 언어가 포함된 경우
    """
    invalid_languages = [lang for lang in languages if lang not in ALLOWED_LANGUAGES]
    
    assert len(invalid_languages) == 0, (
        f"허용되지 않은 언어가 포함되어 있습니다: {invalid_languages}\n"
        f"허용된 언어 목록: {list(ALLOWED_LANGUAGES.keys())}"
    )


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # 1. 언어 검증
    print(f"🔍 입력된 언어: {args.languages}")
    validate_languages(args.languages)
    print(f"✅ 언어 검증 완료")
    
    # 2. 허용된 확장자 목록 생성
    target_extensions = []
    for lang in args.languages:
        target_extensions.extend(ALLOWED_LANGUAGES[lang])
    print(f"📝 대상 확장자: {target_extensions}")
    
    # 3. 데이터셋 로드 (스트리밍 모드)
    print(f"🌐 데이터셋 로드 중...")
    streaming_data = load_dataset(
        "nick007x/github-code-2025", 
        streaming=True
    )
    
    # 4. 필터링하여 데이터 수집
    print(f"🔎 필터링 시작 (목표: {args.sample_size}개)")
    collected_data = []
    total_checked = 0
    
    try:
        for example in streaming_data['train']:
            total_checked += 1
            
            # 파일 경로에서 언어 추론
            if 'file_path' not in example:
                continue
                
            detected_language = get_language_from_filepath(example['file_path'])
            
            # 원하는 언어인 경우 수집
            if detected_language in args.languages:
                # 데이터 구조 유지: repo_id, size, file_path, content
                collected_data.append({
                    'repo_id': example.get('repo_id', 'unknown'),
                    'size': example.get('size', len(example.get('content', ''))),
                    'file_path': example['file_path'],
                    'content': example.get('content', '')
                })
                
                # 진행 상황 출력
                if len(collected_data) % 10 == 0:
                    print(f"  📦 수집됨: {len(collected_data)}/{args.sample_size} (검사: {total_checked}개)")
            
            # 목표 개수 달성 시 종료
            if len(collected_data) >= args.sample_size:
                print(f"✅ 목표 달성! {len(collected_data)}개 수집 완료")
                break
    finally:
        # 스트리밍 데이터 정리 (GIL 관련 에러 방지)
        del streaming_data
    
    # 5. 결과 출력
    print(f"\n📊 수집 결과:")
    print(f"  - 총 검사한 파일: {total_checked}개")
    print(f"  - 수집된 파일: {len(collected_data)}개")
    
    if len(collected_data) < args.sample_size:
        print(f"  ⚠️ 목표({args.sample_size}개)에 미달했지만 가능한 만큼 수집했습니다.")
    
    # 언어별 통계
    language_counts = {}
    for data in collected_data:
        lang = get_language_from_filepath(data['file_path'])
        language_counts[lang] = language_counts.get(lang, 0) + 1
    
    print(f"\n📈 언어별 통계:")
    for lang, count in sorted(language_counts.items()):
        print(f"  - {lang}: {count}개")
    
    # 6. JSON으로 저장
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(collected_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 저장 완료: {args.output_path}")


if __name__ == "__main__":
    main()