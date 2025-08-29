# ───── 표준 라이브러리 ───────────────────────────────
import json
import argparse
from typing import List, Dict
from collections import defaultdict

# ───── 서드파티 라이브러리 ──────────────────────────
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm

# --- 환경 변수 로드 ---
# .env 파일에 OPENAI_API_KEY="your_key" 형식으로 키를 저장해주세요.
load_dotenv()

# --- 모델 초기화 ---
try:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
except Exception as e:
    print(f"[에러] OpenAI 모델 초기화 중 오류 발생: {e}")
    print("OPENAI_API_KEY가 올바르게 설정되었는지 확인해주세요.")
    exit()

def load_sysmon_events(file_path: str) -> List[Dict]:
    """
    Sysmon 이벤트 로그가 담긴 JSON 파일을 로드합니다.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            events = json.load(f)
        if not isinstance(events, list):
            print("[에러] JSON 파일의 최상위 구조는 리스트(Array)여야 합니다.")
            return []
        return sorted(events, key=lambda x: x.get("TimeCreated", ""))
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

def build_process_traces(events: List[Dict]) -> List[Dict]:
    """
    이벤트 리스트를 바탕으로 프로세스 계보를 분석하고 트레이스로 그룹화합니다.
    """
    parent_map = {}
    for event in events:
        if event.get("EventID") == "1":
            child_guid = event.get("ProcessGuid")
            parent_guid = event.get("ParentProcessGuid")
            if child_guid and parent_guid:
                parent_map[child_guid] = parent_guid

    root_cache = {}
    def find_root_process(process_guid: str) -> str:
        if process_guid in root_cache: return root_cache[process_guid]
        parent = parent_map.get(process_guid)
        if parent is None or parent == process_guid:
            root_cache[process_guid] = process_guid
            return process_guid
        root = find_root_process(parent)
        root_cache[process_guid] = root
        return root

    traces_dict = defaultdict(list)
    for event in events:
        process_guid = event.get("ProcessGuid")
        if process_guid:
            root_guid = find_root_process(process_guid)
            traces_dict[root_guid].append(event)
    
    return [{"trace_id": tid, "events": evs} for tid, evs in traces_dict.items()]

def summarize_trace_with_llm(trace: Dict) -> str:
    """
    하나의 트레이스를 입력받아 LLM으로 자연어 요약을 생성합니다.
    """
    # EventID에 따라 핵심 정보만 간결하게 추출
    key_events = []
    for event in trace['events']:
        event_id = event.get("EventID")
        info = {"EventID": event_id, "Image": event.get("Image")}
        if event_id == "1": info.update({"EventType": "ProcessCreate", "ParentImage": event.get("ParentImage"), "CommandLine": event.get("CommandLine")})
        elif event_id == "3": info.update({"EventType": "NetworkConnect", "DestinationIp": event.get("DestinationIp"), "DestinationPort": event.get("DestinationPort")})
        elif event_id == "11": info.update({"EventType": "FileCreate", "TargetFilename": event.get("TargetFilename")})
        # 필요한 다른 EventID에 대한 규칙 추가
        key_events.append({k: v for k, v in info.items() if v})

    events_str = "\n".join([json.dumps(event, ensure_ascii=False) for event in key_events[:30]]) # 너무 길지 않게
    
    prompt = ChatPromptTemplate.from_template(
        """당신은 EDR 로그를 분석하는 최고의 위협 분석가입니다.
        다음은 하나의 트레이스 ID({trace_id})로 묶인 Sysmon 이벤트 시퀀스의 핵심 정보입니다.
        이벤트들의 인과관계를 분석하여 전체적인 행위가 무엇을 의미하는지 전문가 보고서 형식으로 설명해주세요.

        [분석할 이벤트 시퀀스]
        {events}

        [전문가 분석 보고서]
        """
    )
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"trace_id": trace['trace_id'], "events": events_str})
    return summary

def main():
    parser = argparse.ArgumentParser(description="Sysmon 로그를 트레이스로 묶고, 각 트레이스를 자연어로 요약하여 저장합니다.")
    parser.add_argument("--input", type=str, required=True, help="분석할 Sysmon 로그 파일의 경로 (JSON 형식)")
    parser.add_argument("--output", type=str, required=True, help="최종 분석 결과를 저장할 파일 경로 (JSON 형식)")
    args = parser.parse_args()

    print(f"'{args.input}' 파일에서 Sysmon 이벤트를 로드합니다...")
    events = load_sysmon_events(args.input)
    if not events:
        print("분석할 이벤트가 없습니다.")
        return
        
    print("\n프로세스 계보를 분석하여 이벤트를 트레이스로 그룹화합니다...")
    grouped_traces = build_process_traces(events)
    
    print("\n각 트레이스에 대한 자연어 요약을 생성합니다...")
    final_results = []
    for trace in tqdm(grouped_traces, desc="Summarizing Traces"):
        summary = summarize_trace_with_llm(trace)
        trace['summary'] = summary # 기존 트레이스 딕셔너리에 'summary' 키 추가
        final_results.append(trace)

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=4)
        print(f"\n✅ 성공! 총 {len(final_results)}개의 트레이스와 요약 결과를 '{args.output}' 파일에 저장했습니다.")
    except Exception as e:
        print(f"\n[에러] 결과를 파일에 저장하는 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main()