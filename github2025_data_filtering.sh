
#!/bin/bash

# 수집 수량 설정
SAMPLE_SIZE=${1:-1500}  # 첫 번째 인자로 수량 지정, 기본값 1500

echo "🔢 수집 수량: ${SAMPLE_SIZE}개"

# C 데이터 필터링
echo "🔍 C 언어 데이터 필터링 시작..."
python preprocess/filtering_github2025.py --languages c --sample_size ${SAMPLE_SIZE} --output_path data/output/github2025_c.json

# Cpp 필터링
echo "🔍 C++ 언어 데이터 필터링 시작..."
python preprocess/filtering_github2025.py --languages cpp --sample_size ${SAMPLE_SIZE} --output_path data/output/github2025_cpp.json

echo "✅ 모든 필터링 완료!"

# 그외 데이터 필터링 및  수집 
# # Python 데이터 필터링
# python preprocess/filtering_github2025.py --languages python --sample_size 500 --output_path data/output/github2025_c.json
# # JavaScript 데이터 필터링
# python preprocess/filtering_github2025.py --languages javascript --sample_size 500 --output_path data/output/github2025_javascript.json
# # Java 데이터 필터링
# python preprocess/filtering_github2025.py --languages java --sample_size 500 --output_path data/output/github2025_java.json
# # Go 데이터 필터링
# python preprocess/filtering_github2025.py --languages go --sample_size 500 --output_path data/output/github2025_go.json
# # Rust 데이터 필터링
# python preprocess/filtering_github2025.py --languages rust --sample_size 500 --output_path data/output/github2025_rust.json
# # Ruby 데이터 필터링
# python preprocess/filtering_github2025.py --languages ruby --sample_size 500 --output_path data/output/github2025_ruby.json
# # Scala 데이터 필터링
# python preprocess/filtering_github2025.py --languages scala --sample_size 500 --output_path data/output/github2025_scala.json
