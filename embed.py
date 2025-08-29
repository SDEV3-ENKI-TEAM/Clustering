# ───── 표준 라이브러리 ───────────────────────────────
import json
import argparse
from typing import List, Dict

# ───── 서드파티 라이브러리 ──────────────────────────
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np

load_dotenv()

try:
    # 문장/문단 임베딩에 강력한 성능을 보이는 오픈소스 모델을 사용
    print("임베딩 모델(paraphrase-multilingual-mpnet-base-v2)을 로드합니다...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    print("모델 로드 완료.")
except Exception as e:
    print(f"[에러] 임베딩 모델 로딩 중 오류 발생: {e}")
    print("인터넷 연결을 확인하고, 'pip install sentence-transformers'가 실행되었는지 확인해주세요.")
    exit()

def load_summaries_from_file(file_path: str) -> List[Dict]:
    """
    요약 결과가 담긴 JSON 파일을 로드합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            summaries = json.load(f)
        if not isinstance(summaries, list):
            print("[에러] JSON 파일의 최상위 구조는 리스트(Array)여야 합니다.")
            return []
        return summaries
    except FileNotFoundError:
        print(f"[에러] 파일을 찾을 수 없습니다: {file_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"[에러] JSON 파일 형식이 올바르지 않습니다.")
        print(f"  -> 오류 상세: {e.msg} (Line {e.lineno}, Column {e.colno})")
        return []
    except Exception as e:
        print(f"[에러] 파일을 읽는 중 예기치 않은 오류 발생: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="트레이스 요약 파일을 임베딩하여 새로운 파일로 저장합니다.")
    parser.add_argument("--input", type="str", required=True, help="분석할 요약 파일의 경로 (JSON 형식)")
    parser.add_argument("--output", type="str", required=True, help="임베딩 결과를 저장할 파일 경로 (JSON 형식)")
    args = parser.parse_args()

    # --- 1. 파일에서 요약 데이터 로드 ---
    print(f"'{args.input}' 파일에서 요약 데이터를 로드합니다...")
    summaries_to_process = load_summaries_from_file(args.input)
    if not summaries_to_process:
        print("임베딩할 데이터가 없습니다.")
        return

    # --- 2. 각 요약문을 임베딩 ---
    print("\n각 요약문에 대한 임베딩을 시작합니다...")
    
    # 더 빠른 처리를 위해 모든 요약문을 리스트로 만들어 한 번에 인코딩
    summary_texts = [item.get("summary", "") for item in summaries_to_process]
    
    # model.encode는 NumPy 배열을 반환합니다.
    embedding_vectors = embedding_model.encode(summary_texts, show_progress_bar=True)

    # --- 3. 최종 결과를 파일로 저장 ---
    final_results = []
    for i, item in enumerate(summaries_to_process):
        final_results.append({
            "trace_id": item.get("trace_id"),
            "summary": item.get("summary"),
            # JSON 저장을 위해 NumPy 배열을 Python 리스트로 변환
            "embedding": embedding_vectors[i].tolist()
        })

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 성공! 총 {len(final_results)}개의 임베딩 결과를 '{args.output}' 파일에 저장했습니다.")
    except Exception as e:
        print(f"\n[에러] 결과를 파일에 저장하는 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()