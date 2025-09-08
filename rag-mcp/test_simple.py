#!/usr/bin/env python3
"""간단한 MCP 하이브리드 검색 테스트"""

import json
import subprocess
import sys
import asyncio

async def test_mcp_search():
    """MCP 서버에서 하이브리드 검색을 테스트합니다."""
    
    print("🔍 MCP 하이브리드 검색 테스트 시작...")
    
    # MCP 서버 실행
    process = await asyncio.create_subprocess_exec(
        sys.executable, "server.py",
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # 1. 초기화 요청
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": True}}
            }
        }
        
        print("📡 초기화 요청 전송...")
        init_json = json.dumps(init_request) + "\n"
        process.stdin.write(init_json.encode())
        await process.stdin.drain()
        
        # 초기화 응답 대기
        response = await asyncio.wait_for(process.stdout.readline(), timeout=30)
        init_response = json.loads(response.decode().strip())
        print(f"✅ 초기화 완료: {init_response.get('result', {}).get('serverInfo', {}).get('name', 'Unknown')}")
        
        # 2. initialized 알림
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        
        process.stdin.write((json.dumps(initialized_notification) + "\n").encode())
        await process.stdin.drain()
        
        # 잠시 대기
        await asyncio.sleep(1)
        
        # 3. 검색 요청
        search_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {
                    "query": "한국어 자연어 처리와 기계학습",
                    "limit": 3
                }
            }
        }
        
        print("🔍 하이브리드 검색 요청 전송...")
        search_json = json.dumps(search_request) + "\n"
        process.stdin.write(search_json.encode())
        await process.stdin.drain()
        
        # 검색 응답 대기
        search_response_line = await asyncio.wait_for(process.stdout.readline(), timeout=30)
        search_response = json.loads(search_response_line.decode().strip())
        
        print(f"\n🔍 검색 응답 전체:")
        print(json.dumps(search_response, indent=2, ensure_ascii=False))
        
        if "result" in search_response:
            print("✅ 검색 성공!")
            result = search_response["result"]
            if result and len(result) > 0:
                if isinstance(result[0], dict) and "text" in result[0]:
                    result_text = result[0]["text"]
                    print("\n📊 검색 결과:")
                    print("-" * 60)
                    print(result_text)
                    print("-" * 60)
                else:
                    print("⚠️ 결과 형식이 예상과 다릅니다.")
                    print(f"result[0]: {result[0]}")
        else:
            print("❌ 검색 실패!")
            if "error" in search_response:
                print(f"오류: {search_response['error']}")
        
    except asyncio.TimeoutError:
        print("❌ 응답 타임아웃")
    except Exception as e:
        print(f"❌ 오류: {e}")
        
        # stderr 출력 확인
        try:
            stderr_output = await asyncio.wait_for(process.stderr.read(), timeout=1)
            if stderr_output:
                print(f"서버 오류 로그:\n{stderr_output.decode()}")
        except:
            pass
            
    finally:
        process.terminate()
        await process.wait()

if __name__ == "__main__":
    asyncio.run(test_mcp_search())