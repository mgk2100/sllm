#!/bin/bash

# SFT 데이터 생성 실행 예시 스크립트
# 사용법: bash run_sft_example.sh

# ==================================================
# 설정 (본인의 환경에 맞게 수정하세요)
# ==================================================

# Azure OpenAI 설정
AZURE_ENDPOINT="https://your-resource-name.openai.azure.com"
AZURE_DEPLOYMENT="gpt-4"  # 또는 "gpt-4-turbo", "gpt-35-turbo" 등
API_VERSION="2024-02-15-preview"

# 파일 경로
INPUT_JSON="data/output/app_code.json"
OUTPUT_JSON="data/output/sft_dataset.json"

# 처리 옵션
MAX_CODE_LENGTH=2000       # 처리할 최대 코드 길이
STRATEGIES_PER_CODE=3      # 각 코드당 적용할 전략 개수 (1-8)
SAMPLE_SIZE=""             # 테스트용 샘플 크기 (빈 문자열이면 전체 사용)

# ==================================================
# 실행
# ==================================================

echo "🚀 SFT 데이터 생성 시작..."
echo ""
echo "📋 설정:"
echo "  - 입력: $INPUT_JSON"
echo "  - 출력: $OUTPUT_JSON"
echo "  - Azure Endpoint: $AZURE_ENDPOINT"
echo "  - 배포: $AZURE_DEPLOYMENT"
echo "  - 전략 개수: $STRATEGIES_PER_CODE"
echo ""

# 커맨드 구성
CMD="python preprocess/sft_data_generate.py \
  --input_json $INPUT_JSON \
  --output_json $OUTPUT_JSON \
  --azure_endpoint $AZURE_ENDPOINT \
  --azure_deployment $AZURE_DEPLOYMENT \
  --api_version $API_VERSION \
  --max_code_length $MAX_CODE_LENGTH \
  --strategies_per_code $STRATEGIES_PER_CODE"

# 샘플 크기가 설정되어 있으면 추가
if [ -n "$SAMPLE_SIZE" ]; then
  CMD="$CMD --sample_size $SAMPLE_SIZE"
  echo "🔬 테스트 모드: $SAMPLE_SIZE 개 샘플만 사용"
  echo ""
fi

# 실행
eval $CMD

# 결과 확인
if [ $? -eq 0 ]; then
  echo ""
  echo "✅ 성공적으로 완료되었습니다!"
  echo "📁 출력 파일: $OUTPUT_JSON"
else
  echo ""
  echo "❌ 에러가 발생했습니다."
  exit 1
fi

