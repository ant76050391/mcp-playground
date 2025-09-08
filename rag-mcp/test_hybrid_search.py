#!/usr/bin/env python3
"""3단계 하이브리드 검색 테스트 스크립트"""

import json
import subprocess
import sys
import asyncio
import time

async def test_hybrid_search():
    """하이브리드 검색 기능을 테스트합니다."""
    
    print("🔍 3단계 하이브리드 검색 테스트 시작...")
    print("=" * 60)
    
    try:
        # MCP 서버 프로세스 시작
        process = await asyncio.create_subprocess_exec(
            sys.executable, "server.py",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 서버 초기화 대기
        print("⏳ 서버 초기화 대기 중...")
        await asyncio.sleep(3)
        
        # 테스트 쿼리들
        test_queries = [
            {
                "query": "한국어 자연어 처리",
                "description": "한국어 NLP 관련 검색",
                "limit": 3
            },
            {
                "query": "BGE 모델 특징",
                "description": "BGE 모델 기능 검색",
                "limit": 3
            },
            {
                "query": "BM25 검색 알고리즘",
                "description": "BM25 키워드 검색",
                "limit": 3
            },
            {
                "query": "임베딩과 검색",
                "description": "임베딩 기반 검색",
                "limit": 2
            }
        ]
        
        for i, test_case in enumerate(test_queries, 1):
            print(f"\n🔍 테스트 {i}: {test_case['description']}")
            print(f"   쿼리: '{test_case['query']}'")
            
            # MCP 검색 요청 생성
            search_request = {
                "jsonrpc": "2.0",
                "id": i,
                "method": "tools/call",
                "params": {
                    "name": "search_documents",
                    "arguments": {
                        "query": test_case["query"],
                        "limit": test_case["limit"]
                    }
                }
            }
            
            # 요청 전송
            request_str = json.dumps(search_request) + "\n"
            process.stdin.write(request_str.encode())
            await process.stdin.drain()
            
            # 응답 읽기
            try:
                response_line = await asyncio.wait_for(
                    process.stdout.readline(), 
                    timeout=30.0
                )
                
                if response_line:
                    response = json.loads(response_line.decode().strip())
                    
                    if "result" in response and "content" in response["result"][0]:
                        result_text = response["result"][0]["content"]
                        print("   결과:")
                        # 결과를 보기 좋게 포맷팅
                        lines = result_text.split('\n')
                        for line in lines[:20]:  # 처음 20줄만 표시
                            if line.strip():
                                print(f"     {line}")
                        if len(lines) > 20:
                            print("     ...")
                    else:
                        print("   ❌ 검색 결과를 파싱할 수 없습니다")
                        print(f"   응답: {response}")
                        
            except asyncio.TimeoutError:
                print("   ❌ 응답 타임아웃 (30초)")
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON 파싱 오류: {e}")
            except Exception as e:
                print(f"   ❌ 오류: {e}")
            
            # 테스트 간 잠시 대기
            await asyncio.sleep(1)
        
        print(f"\n✅ 모든 테스트 완료!")
        
    except Exception as e:
        print(f"❌ 테스트 실행 오류: {e}")
    finally:
        # 프로세스 종료
        if process:
            process.terminate()
            await process.wait()

if __name__ == "__main__":
    print("🚀 3단계 하이브리드 검색 (Dense + Sparse + Reranking) 테스트")
    print("📋 구성: BGE-M3 + BM25 + BGE-Reranker-v2-M3-Ko")
    print("=" * 60)
    
    try:
        asyncio.run(test_hybrid_search())
    except KeyboardInterrupt:
        print("\n⏹️  테스트 중단됨")
    except Exception as e:
        print(f"❌ 실행 오류: {e}")