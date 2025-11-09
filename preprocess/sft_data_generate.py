"""
SFT 데이터 생성 스크립트
코드 파일을 읽어 Azure OpenAI를 활용하여 instruction-output 쌍을 생성합니다.

사용 예시:
python sft_data_generate.py --input_json data/output/app_code.json --output_json data/output/sft_dataset.json
"""

import argparse
import json
import os
from typing import List, Dict, Any
from tqdm import tqdm
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


class SFTDataGenerator:
    """
    코드 데이터를 SFT 학습용 instruction-output 쌍으로 변환하는 클래스
    """
    
    def __init__(self, llm: AzureChatOpenAI, max_code_length: int = 2000):
        """
        Args:
            llm: Azure OpenAI LLM 인스턴스
            max_code_length: 처리할 최대 코드 길이 (너무 긴 코드는 스킵)
        """
        self.llm = llm
        self.max_code_length = max_code_length
        
        # 다양한 instruction 생성 전략들
        self.strategies = [
            #self._generate_code_explanation,
            self._generate_code_documentation,
            #self._generate_code_improvement,
            self._generate_code_completion,
            self._generate_function_implementation,
            #self._generate_bug_detection,
            #self._generate_code_refactoring,
            self._generate_code_summary,
        ]
    
    def _create_prompt(self, system_message: str, user_message: str) -> List:
        """프롬프트 메시지 생성"""
        return [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
    
    def _generate_code_explanation(self, code_entry: Dict[str, Any]) -> Dict[str, str]:
        """
        전략 1: 코드 설명 생성
        코드를 보고 상세한 설명을 생성합니다.
        """
        code = code_entry["content"]
        file_path = code_entry["file_path"]
        
        system_msg = "당신은 코드를 분석하고 명확하게 설명하는 전문가입니다."
        user_msg = f"""다음 코드 파일({file_path})을 분석하고 한국어로 상세히 설명해주세요:

```
{code}
```

다음 항목을 포함해서 설명해주세요:
1. 코드의 주요 목적과 기능
2. 주요 함수/클래스의 역할
3. 중요한 로직이나 알고리즘
4. 사용된 라이브러리나 의존성"""

        messages = self._create_prompt(system_msg, user_msg)
        response = self.llm.invoke(messages)
        
        return {
            "instruction": f"다음 코드를 분석하고 상세히 설명해주세요:\n\n```\n{code}\n```",
            "output": response.content,
            "metadata": {
                "strategy": "code_explanation",
                "repo_id": code_entry["repo_id"],
                "file_path": file_path
            }
        }
    
    def _generate_code_documentation(self, code_entry: Dict[str, Any]) -> Dict[str, str]:
        """
        전략 2: 코드 문서화
        코드에 주석과 docstring을 추가합니다.
        """
        code = code_entry["content"]
        file_path = code_entry["file_path"]
        
        system_msg = "당신은 코드 문서화 전문가입니다. 가독성 높은 주석과 docstring을 작성합니다."
        user_msg = f"""다음 코드에 적절한 주석과 docstring을 추가해주세요:

```
{code}
```

- 함수/클래스에는 docstring 추가
- 복잡한 로직에는 inline 주석 추가
- 한국어로 작성"""

        messages = self._create_prompt(system_msg, user_msg)
        response = self.llm.invoke(messages)
        
        return {
            "instruction": "다음 코드에 주석과 docstring을 추가해주세요:\n\n```\n" + code + "\n```",
            "output": response.content,
            "metadata": {
                "strategy": "code_documentation",
                "repo_id": code_entry["repo_id"],
                "file_path": file_path
            }
        }
    
    def _generate_code_improvement(self, code_entry: Dict[str, Any]) -> Dict[str, str]:
        """
        전략 3: 코드 개선
        코드를 분석하고 개선 방안을 제시합니다.
        """
        code = code_entry["content"]
        file_path = code_entry["file_path"]
        
        system_msg = "당신은 코드 리뷰와 개선을 전문으로 하는 시니어 개발자입니다."
        user_msg = f"""다음 코드를 분석하고 개선 방안을 제시해주세요:

```
{code}
```

다음 관점에서 분석해주세요:
1. 성능 최적화 가능 부분
2. 가독성 개선
3. 에러 처리 개선
4. 베스트 프랙티스 적용
5. 개선된 코드 제시"""

        messages = self._create_prompt(system_msg, user_msg)
        response = self.llm.invoke(messages)
        
        return {
            "instruction": "다음 코드를 분석하고 개선 방안을 제시해주세요:\n\n```\n" + code + "\n```",
            "output": response.content,
            "metadata": {
                "strategy": "code_improvement",
                "repo_id": code_entry["repo_id"],
                "file_path": file_path
            }
        }
    
    def _generate_code_completion(self, code_entry: Dict[str, Any]) -> Dict[str, str]:
        """
        전략 4: 코드 완성
        코드의 일부를 제거하고 완성하도록 합니다.
        """
        code = code_entry["content"]
        file_path = code_entry["file_path"]
        
        # 코드를 대략 70% 정도만 사용 (간단한 전략)
        lines = code.split('\n')
        cutoff = int(len(lines) * 0.7)
        partial_code = '\n'.join(lines[:cutoff])
        
        system_msg = "당신은 코드 완성을 돕는 AI 어시스턴트입니다."
        user_msg = f"""다음 미완성 코드를 분석하고 나머지 부분을 완성해주세요:

```
{partial_code}
```

원본 파일: {file_path}
논리적 흐름을 유지하면서 코드를 완성해주세요."""

        messages = self._create_prompt(system_msg, user_msg)
        response = self.llm.invoke(messages)
        
        return {
            "instruction": f"다음 미완성 코드를 완성해주세요:\n\n```\n{partial_code}\n```",
            "output": response.content,
            "metadata": {
                "strategy": "code_completion",
                "repo_id": code_entry["repo_id"],
                "file_path": file_path
            }
        }
    
    def _generate_function_implementation(self, code_entry: Dict[str, Any]) -> Dict[str, str]:
        """
        전략 5: 자연어 설명으로부터 코드 생성
        코드를 먼저 설명으로 변환한 후, 그 설명으로부터 코드를 생성하는 쌍을 만듭니다.
        """
        code = code_entry["content"]
        file_path = code_entry["file_path"]
        
        # 파일 확장자로 언어 추론
        file_ext = file_path.split('.')[-1].lower()
        lang_map = {
            'py': 'python',
            'js': 'javascript',
            'ts': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c': 'c',
            'go': 'go',
            'rs': 'rust',
            'rb': 'ruby',
            'php': 'php',
        }
        language = lang_map.get(file_ext, '')
        
        # Step 1: 코드를 자연어 요구사항으로 변환
        system_msg1 = "당신은 코드를 자연어 요구사항으로 변환하는 전문가입니다."
        user_msg1 = f"""다음 코드를 자연어 요구사항으로 변환해주세요. 마치 개발자에게 구현을 요청하는 것처럼 작성해주세요:

```
{code}
```

형식: "다음과 같은 기능을 구현해주세요: ..." 스타일로 작성"""

        messages1 = self._create_prompt(system_msg1, user_msg1)
        requirement = self.llm.invoke(messages1).content
        
        # 코드를 코드 블록으로 감싸기
        output_with_codeblock = f"```{language}\n{code}\n```"
        
        return {
            "instruction": requirement,
            "output": output_with_codeblock,
            "metadata": {
                "strategy": "function_implementation",
                "repo_id": code_entry["repo_id"],
                "file_path": file_path
            }
        }
    
    def _generate_bug_detection(self, code_entry: Dict[str, Any]) -> Dict[str, str]:
        """
        전략 6: 버그 탐지 및 수정
        코드의 잠재적 버그나 문제점을 찾습니다.
        """
        code = code_entry["content"]
        file_path = code_entry["file_path"]
        
        system_msg = "당신은 코드의 버그와 보안 취약점을 찾는 전문가입니다."
        user_msg = f"""다음 코드를 분석하고 잠재적인 버그, 에러, 보안 취약점을 찾아주세요:

```
{code}
```

다음을 포함해주세요:
1. 발견된 문제점들
2. 각 문제의 심각도
3. 수정 방법
4. 수정된 코드"""

        messages = self._create_prompt(system_msg, user_msg)
        response = self.llm.invoke(messages)
        
        return {
            "instruction": f"다음 코드에서 버그나 문제점을 찾고 수정해주세요:\n\n```\n{code}\n```",
            "output": response.content,
            "metadata": {
                "strategy": "bug_detection",
                "repo_id": code_entry["repo_id"],
                "file_path": file_path
            }
        }
    
    def _generate_code_refactoring(self, code_entry: Dict[str, Any]) -> Dict[str, str]:
        """
        전략 7: 코드 리팩토링
        코드를 더 나은 구조로 리팩토링합니다.
        """
        code = code_entry["content"]
        file_path = code_entry["file_path"]
        
        system_msg = "당신은 클린 코드와 리팩토링 전문가입니다."
        user_msg = f"""다음 코드를 리팩토링해주세요:

```
{code}
```

리팩토링 원칙:
1. DRY (Don't Repeat Yourself) 원칙 적용
2. 함수/변수명 개선
3. 코드 구조 개선
4. 디자인 패턴 적용 (필요시)
5. 리팩토링 전후 비교 설명"""

        messages = self._create_prompt(system_msg, user_msg)
        response = self.llm.invoke(messages)
        
        return {
            "instruction": f"다음 코드를 클린 코드 원칙에 따라 리팩토링해주세요:\n\n```\n{code}\n```",
            "output": response.content,
            "metadata": {
                "strategy": "code_refactoring",
                "repo_id": code_entry["repo_id"],
                "file_path": file_path
            }
        }
    
    def _generate_code_summary(self, code_entry: Dict[str, Any]) -> Dict[str, str]:
        """
        전략 8: 코드 요약
        코드의 핵심 내용을 간결하게 요약합니다.
        """
        code = code_entry["content"]
        file_path = code_entry["file_path"]
        
        system_msg = "당신은 코드를 간결하고 명확하게 요약하는 전문가입니다."
        user_msg = f"""다음 코드를 간결하게 요약해주세요 (3-5문장):

```
{code}
```

포함할 내용:
- 주요 기능
- 핵심 로직
- 입출력"""

        messages = self._create_prompt(system_msg, user_msg)
        response = self.llm.invoke(messages)
        
        return {
            "instruction": f"다음 코드를 간결하게 요약해주세요:\n\n```\n{code}\n```",
            "output": response.content,
            "metadata": {
                "strategy": "code_summary",
                "repo_id": code_entry["repo_id"],
                "file_path": file_path
            }
        }
    
    def generate_sft_data(
        self, 
        code_entries: List[Dict[str, Any]], 
        strategies_per_code: int = 3,
        skip_errors: bool = True
    ) -> List[Dict[str, str]]:
        """
        코드 데이터를 SFT 데이터로 변환합니다.
        
        Args:
            code_entries: 코드 데이터 리스트 (repo_id, file_path, content, size 포함)
            strategies_per_code: 각 코드당 적용할 전략 개수
            skip_errors: 에러 발생시 스킵할지 여부
            
        Returns:
            SFT 데이터셋 (instruction, output, metadata 포함)
        """
        sft_dataset = []
        
        for code_entry in tqdm(code_entries, desc="SFT 데이터 생성 중"):
            # 코드가 너무 길면 스킵
            if len(code_entry["content"]) > self.max_code_length:
                print(f"⚠️ 코드가 너무 깁니다 (길이: {len(code_entry['content'])}). 스킵: {code_entry['file_path']}")
                continue
            
            # 선택된 전략들을 적용
            selected_strategies = self.strategies[:strategies_per_code]
            
            for strategy_func in selected_strategies:
                try:
                    sft_entry = strategy_func(code_entry)
                    sft_dataset.append(sft_entry)
                except Exception as e:
                    error_msg = f"❌ 에러 발생 ({strategy_func.__name__}): {code_entry['file_path']}\n{str(e)}"
                    if skip_errors:
                        print(error_msg)
                        continue
                    else:
                        raise Exception(error_msg)
        
        return sft_dataset


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description="코드 데이터를 SFT 학습용 데이터로 변환")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="입력 JSON 파일 경로 (make_autosar_cpt_data.py 출력 형식)"
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="출력 JSON 파일 경로"
    )
    parser.add_argument(
        "--azure_endpoint",
        type=str,
        required=True,
        help="Azure OpenAI 엔드포인트"
    )
    parser.add_argument(
        "--azure_deployment",
        type=str,
        required=True,
        help="Azure OpenAI 배포 이름"
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default="2024-02-15-preview",
        help="Azure OpenAI API 버전"
    )
    parser.add_argument(
        "--max_code_length",
        type=int,
        default=2000,
        help="처리할 최대 코드 길이"
    )
    parser.add_argument(
        "--strategies_per_code",
        type=int,
        default=3,
        help="각 코드당 적용할 전략 개수 (1-8)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="샘플링할 코드 개수 (테스트용, None이면 전체 사용)"
    )
    
    return parser.parse_args()


def main():
    """메인 실행 함수"""
    args = parse_args()
    
    # 1. 입력 JSON 파일 로드
    print(f"📂 입력 파일 로딩: {args.input_json}")
    with open(args.input_json, "r", encoding="utf-8") as f:
        code_entries = json.load(f)
    
    print(f"✅ 총 {len(code_entries)}개의 코드 파일 로드됨")
    
    # 샘플링 (테스트용)
    if args.sample_size:
        code_entries = code_entries[:args.sample_size]
        print(f"🔬 샘플링: {args.sample_size}개만 사용")
    
    # 2. Azure OpenAI LLM 초기화
    print(f"🤖 Azure OpenAI 초기화 중...")
    llm = AzureChatOpenAI(
        azure_endpoint=args.azure_endpoint,
        azure_deployment=args.azure_deployment,
        api_version=args.api_version,
        temperature=0.7,
        max_tokens=2000,
    )
    
    # 3. SFT 데이터 생성기 초기화
    generator = SFTDataGenerator(
        llm=llm,
        max_code_length=args.max_code_length
    )
    
    # 4. SFT 데이터 생성
    print(f"🔧 SFT 데이터 생성 시작...")
    sft_dataset = generator.generate_sft_data(
        code_entries=code_entries,
        strategies_per_code=args.strategies_per_code,
        skip_errors=True
    )
    
    # 5. 결과 저장
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(sft_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 완료!")
    print(f"📊 생성된 SFT 데이터: {len(sft_dataset)}개")
    print(f"💾 저장 위치: {args.output_json}")
    
    # 전략별 통계
    strategy_counts = {}
    for entry in sft_dataset:
        strategy = entry["metadata"]["strategy"]
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    print(f"\n📈 전략별 통계:")
    for strategy, count in sorted(strategy_counts.items()):
        print(f"  - {strategy}: {count}개")


if __name__ == "__main__":
    main()
