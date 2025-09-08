#!/usr/bin/env python3
"""Test BGE embedding functionality via MCP."""

import asyncio
import json
import subprocess
import sys

async def test_embedding_search():
    """Test the embedding functionality through search_documents tool."""
    try:
        # Start the server process
        process = await asyncio.create_subprocess_exec(
            sys.executable, "server.py",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server initialization (embedding model loading)
        print("⏳ Starting server and loading embedding model...")
        await asyncio.sleep(10)  # Give time for model loading
        
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {"listChanged": True},
                    "sampling": {}
                },
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }
        
        # Send the request
        request_str = json.dumps(init_request) + "\n"
        process.stdin.write(request_str.encode())
        await process.stdin.drain()
        
        # Read initialization response
        response = await process.stdout.readline()
        init_response = json.loads(response.decode().strip())
        print(f"✅ Server initialized: {init_response['result']['serverInfo']['name']}")
        
        # Test search_documents tool with Korean query
        search_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {
                    "query": "안녕하세요 한국어 문서 검색",
                    "limit": 3
                }
            }
        }
        
        print("🔍 Testing Korean text embedding...")
        request_str = json.dumps(search_request) + "\n"
        process.stdin.write(request_str.encode())
        await process.stdin.drain()
        
        # Read search response
        response = await process.stdout.readline()
        search_response = json.loads(response.decode().strip())
        
        if 'result' in search_response:
            result_text = search_response['result']['content'][0]['text']
            print(f"✅ Search successful!")
            print(f"📊 Result: {result_text}")
        else:
            print(f"❌ Search failed: {search_response}")
        
        # Test with English query
        search_request_en = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {
                    "query": "Hello English document search",
                    "limit": 3
                }
            }
        }
        
        print("🔍 Testing English text embedding...")
        request_str = json.dumps(search_request_en) + "\n"
        process.stdin.write(request_str.encode())
        await process.stdin.drain()
        
        # Read search response
        response = await process.stdout.readline()
        search_response = json.loads(response.decode().strip())
        
        if 'result' in search_response:
            result_text = search_response['result']['content'][0]['text']
            print(f"✅ English search successful!")
            print(f"📊 Result: {result_text}")
        else:
            print(f"❌ English search failed: {search_response}")
        
        # Clean up
        process.terminate()
        await process.wait()
        
        print("🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if 'process' in locals():
            process.terminate()

if __name__ == "__main__":
    asyncio.run(test_embedding_search())