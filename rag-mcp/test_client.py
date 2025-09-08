#!/usr/bin/env python3
"""Simple MCP client test."""

import asyncio
import json
import subprocess
import sys

async def test_mcp_server():
    """Test the MCP server with a simple request."""
    try:
        # Start the server process
        process = await asyncio.create_subprocess_exec(
            sys.executable, "server.py",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Send initialization request
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "roots": {
                        "listChanged": True
                    },
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
        
        # Read response with timeout
        try:
            stdout_data = await asyncio.wait_for(
                process.stdout.readline(), 
                timeout=5.0
            )
            response = stdout_data.decode().strip()
            print(f"Server response: {response}")
            
            if response:
                response_data = json.loads(response)
                print(f"✅ Server responded successfully: {response_data}")
            else:
                print("❌ No response from server")
                
        except asyncio.TimeoutError:
            print("❌ Server response timeout")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON response: {e}")
        
        # Clean up
        process.terminate()
        await process.wait()
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())