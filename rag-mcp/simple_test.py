#!/usr/bin/env python3
"""Simple test for BGE embedding functionality."""

import asyncio
import json
import subprocess
import sys

async def simple_test():
    """Simple test of MCP server with tools list."""
    try:
        # Start the server process
        process = await asyncio.create_subprocess_exec(
            sys.executable, "server.py",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server initialization
        print("⏳ Starting server...")
        await asyncio.sleep(10)
        
        # Send initialization
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        process.stdin.write((json.dumps(init_request) + "\n").encode())
        await process.stdin.drain()
        
        response = await process.stdout.readline()
        print(f"Init: {response.decode().strip()}")
        
        # List tools
        tools_request = {
            "jsonrpc": "2.0", 
            "id": 2,
            "method": "tools/list"
        }
        
        process.stdin.write((json.dumps(tools_request) + "\n").encode())
        await process.stdin.drain()
        
        response = await process.stdout.readline()
        tools_response = json.loads(response.decode().strip())
        print(f"✅ Tools available: {[tool['name'] for tool in tools_response['result']['tools']]}")
        
        # Call search tool
        call_request = {
            "jsonrpc": "2.0",
            "id": 3, 
            "method": "tools/call",
            "params": {
                "name": "search_documents",
                "arguments": {"query": "안녕하세요", "limit": 2}
            }
        }
        
        print("🔍 Testing search with Korean text...")
        process.stdin.write((json.dumps(call_request) + "\n").encode())
        await process.stdin.drain()
        
        response = await process.stdout.readline()
        call_response = json.loads(response.decode().strip())
        
        if 'result' in call_response:
            print(f"✅ Search successful!")
            print(f"📊 {call_response['result']['content'][0]['text'][:200]}...")
        else:
            print(f"❌ Search failed: {call_response}")
        
        process.terminate()
        await process.wait()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'process' in locals():
            process.terminate()

if __name__ == "__main__":
    asyncio.run(simple_test())