#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
PYTHON_FILE="$SCRIPT_DIR/server.py"
MODEL_DIR="$SCRIPT_DIR/.model"
MODEL_FILE="$MODEL_DIR/bge-m3-korean-q4_k_m-2.gguf"
RERANKER_MODEL_FILE="$MODEL_DIR/bge-reranker-v2-m3-ko-q4_k_m.gguf"
VECTORDB_DIR="$SCRIPT_DIR/.vectordb"
VECTORDB_CHECK_FILE="$VECTORDB_DIR/.chroma_initialized"
DOCUMENTS_DIR="$SCRIPT_DIR/documents"
CONFIG_FILE="$SCRIPT_DIR/config.json"

echo "RAG MCP Server Installer & Runner"
echo "================================"

# Display system information
echo "System Information:"
echo "- OS: $(uname -s) $(uname -r)"
echo "- Architecture: $(uname -m)"
echo "- CPU cores: $(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'Unknown')"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "- GPU: $(system_profiler SPDisplaysDataType 2>/dev/null | grep 'Chipset Model:' | head -1 | cut -d: -f2 | xargs || echo 'Unknown')"
fi

# Check and setup Java for KoNLPy compatibility
echo ""
echo "Setting up Java for Korean tokenizer (KoNLPy):"

# Check if OpenJDK 11 is already installed via Homebrew
JAVA11_INSTALLED=false
if [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null; then
    if brew list openjdk@11 &> /dev/null; then
        JAVA11_INSTALLED=true
        echo "✅ OpenJDK 11 already installed via Homebrew"
    fi
fi

# Install OpenJDK 11 if not present and on macOS with Homebrew
if [ "$JAVA11_INSTALLED" = false ] && [[ "$OSTYPE" == "darwin"* ]] && command -v brew &> /dev/null; then
    echo "📦 Installing OpenJDK 11 for KoNLPy compatibility..."
    if brew install openjdk@11; then
        echo "✅ OpenJDK 11 installed successfully"
        JAVA11_INSTALLED=true
    else
        echo "❌ Failed to install OpenJDK 11"
    fi
fi

# Setup JAVA_HOME for this session if OpenJDK 11 is available
if [ "$JAVA11_INSTALLED" = true ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Apple Silicon or Intel Mac paths
        if [ -d "/opt/homebrew/opt/openjdk@11" ]; then
            export JAVA_HOME="/opt/homebrew/opt/openjdk@11"
        elif [ -d "/usr/local/opt/openjdk@11" ]; then
            export JAVA_HOME="/usr/local/opt/openjdk@11"
        fi
        
        if [ -n "$JAVA_HOME" ]; then
            export PATH="$JAVA_HOME/bin:$PATH"
            echo "🔧 JAVA_HOME set to: $JAVA_HOME"
            echo "- Java version: $(java -version 2>&1 | head -n 1)"
            echo "✅ Korean BM25 tokenizer will use KoNLPy with proper Java version"
        fi
    fi
else
    # Check current Java version
    if command -v java &> /dev/null; then
        JAVA_VERSION=$(java -version 2>&1 | head -n 1 | cut -d'"' -f2 | cut -d'.' -f1)
        echo "- Current Java version: $(java -version 2>&1 | head -n 1)"
        
        if [ "$JAVA_VERSION" -ge 17 ]; then
            echo "⚠️  Java $JAVA_VERSION may have compatibility issues with KoNLPy"
            echo "   Korean BM25 tokenizer will fall back to regex-based tokenization"
        fi
    else
        echo "⚠️  Java not found. Korean BM25 will use regex tokenizer"
    fi
fi
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Please install uv first."
    echo "Visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment with uv..."
    uv venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Function to detect GPU capabilities
detect_gpu_support() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # Check for Apple Silicon or Intel Mac with dedicated GPU
        if system_profiler SPHardwareDataType 2>/dev/null | grep -q "Apple M\|Apple"; then
            echo "metal"  # Apple Silicon with integrated GPU
        elif system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
            echo "metal"  # Intel Mac with Metal-capable GPU
        else
            echo "cpu"     # Fallback to CPU
        fi
    elif command -v nvidia-smi &> /dev/null; then
        echo "cuda"       # NVIDIA GPU detected
    elif lspci 2>/dev/null | grep -i amd | grep -i vga &> /dev/null; then
        echo "rocm"       # AMD GPU detected
    else
        echo "cpu"        # No GPU detected
    fi
}

# Function to check GPU acceleration support
check_gpu_acceleration() {
    echo "Checking GPU acceleration support..."
    
    # Detect GPU capabilities
    GPU_TYPE=$(detect_gpu_support)
    echo "Detected GPU type: $GPU_TYPE"
    
    # Check if llama-cpp-python is already installed
    if python -c "import llama_cpp; print(f'llama-cpp-python version: {llama_cpp.__version__}')" 2>/dev/null; then
        echo "llama-cpp-python is already installed"
        
        # Test if GPU acceleration is actually working
        if [[ "$GPU_TYPE" == "metal" ]]; then
            # Simple test for Metal support
            if python -c "
import llama_cpp
try:
    # Try to load a minimal model config with GPU layers
    model = llama_cpp.Llama(model_path='$MODEL_FILE', n_gpu_layers=1, n_ctx=64, embedding=True, verbose=False)
    print('Metal GPU acceleration: Working')
    del model
except Exception as e:
    print(f'Metal GPU acceleration: Not working - {e}')
" 2>/dev/null | grep -q "Working"; then
                echo "GPU acceleration is properly configured and working"
                return 0
            else
                echo "GPU acceleration may not be working properly"
            fi
        else
            echo "GPU acceleration status: Platform dependent"
        fi
    else
        echo "llama-cpp-python: Not installed"
    fi
    
    # Install appropriate version based on detected GPU
    echo ""
    case "$GPU_TYPE" in
        "metal")
            echo "Installing llama-cpp-python with Metal GPU support..."
            echo "This may take several minutes to compile..."
            CMAKE_ARGS="-DGGML_METAL=on" uv pip install llama-cpp-python --no-cache-dir
            ;;
        "cuda")
            echo "Installing llama-cpp-python with CUDA support..."
            echo "This may take several minutes to compile..."
            CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir
            ;;
        "rocm")
            echo "Installing llama-cpp-python with ROCm support..."
            echo "This may take several minutes to compile..."
            CMAKE_ARGS="-DGGML_HIPBLAS=on" uv pip install llama-cpp-python --no-cache-dir
            ;;
        *)
            echo "Installing CPU-only version of llama-cpp-python..."
            uv pip install llama-cpp-python --no-cache-dir
            ;;
    esac
    
    if [ $? -eq 0 ]; then
        echo "Installation completed successfully"
    else
        echo "Installation failed. Trying CPU-only fallback..."
        uv pip install llama-cpp-python --no-cache-dir
    fi
}

# Install dependencies if requirements.txt exists
if [ -f "$SCRIPT_DIR/requirements.txt" ]; then
    echo "Installing dependencies..."
    uv pip install -r "$SCRIPT_DIR/requirements.txt"
    
    # Check and setup GPU acceleration after installing dependencies
    check_gpu_acceleration
fi

# Setup model directory and download model if needed
echo "Checking model setup..."
if [ ! -d "$MODEL_DIR" ]; then
    echo "Creating .model directory..."
    mkdir -p "$MODEL_DIR"
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "Looking for bge-m3-korean-q4_k_m-2.gguf in current directory..."
    
    # Check if model file exists in current directory
    if [ -f "$SCRIPT_DIR/bge-m3-korean-q4_k_m-2.gguf" ]; then
        echo "Moving bge-m3-korean-q4_k_m-2.gguf to .model directory..."
        mv "$SCRIPT_DIR/bge-m3-korean-q4_k_m-2.gguf" "$MODEL_FILE"
        echo "✅ Model moved successfully!"
    else
        echo "❌ bge-m3-korean-q4_k_m-2.gguf not found in current directory."
        echo "Please place the model file in: $SCRIPT_DIR/"
        echo "The file will be automatically moved to: $MODEL_FILE"
        exit 1
    fi
else
    echo "✅ Model already exists, skipping setup."
fi

# Setup reranker model
echo "Checking reranker model setup..."
if [ ! -f "$RERANKER_MODEL_FILE" ]; then
    echo "Looking for bge-reranker-v2-m3-ko-q4_k_m.gguf in current directory..."
    
    # Check if reranker model file exists in current directory
    if [ -f "$SCRIPT_DIR/bge-reranker-v2-m3-ko-q4_k_m.gguf" ]; then
        echo "Moving bge-reranker-v2-m3-ko-q4_k_m.gguf to .model directory..."
        mv "$SCRIPT_DIR/bge-reranker-v2-m3-ko-q4_k_m.gguf" "$RERANKER_MODEL_FILE"
        echo "✅ Reranker model moved successfully!"
    else
        echo "⚠️  bge-reranker-v2-m3-ko-q4_k_m.gguf not found in current directory."
        echo "📄 Reranker model is optional - continuing without it."
        echo "📍 If you want to use the reranker, place the model file in: $SCRIPT_DIR/"
        echo "📁 It will be automatically moved to: $RERANKER_MODEL_FILE"
    fi
else
    echo "✅ Reranker model already exists, skipping setup."
fi

# Setup vector database
echo "Checking vector database setup..."
if [ ! -d "$VECTORDB_DIR" ]; then
    echo "Creating .vectordb directory..."
    mkdir -p "$VECTORDB_DIR"
fi

# Function to test ChromaDB health
test_vectordb_health() {
    python3 -c "
import sys
import os
sys.path.insert(0, '$SCRIPT_DIR')
try:
    import chromadb
    client = chromadb.PersistentClient(path='$VECTORDB_DIR')
    # Try to create or get a test collection
    collection = client.get_or_create_collection('health_check')
    print('Vector DB health check passed')
    exit(0)
except Exception as e:
    print(f'Vector DB health check failed: {e}')
    exit(1)
" 2>/dev/null
}

if [ ! -f "$VECTORDB_CHECK_FILE" ]; then
    echo "Initializing ChromaDB vector database..."
    if test_vectordb_health; then
        touch "$VECTORDB_CHECK_FILE"
        echo "✅ Vector database initialized successfully!"
    else
        echo "❌ Vector database initialization failed!"
        exit 1
    fi
else
    echo "Vector database exists, checking health..."
    if test_vectordb_health; then
        echo "✅ Vector database health check passed."
    else
        echo "⚠️  Vector database corrupted, reinitializing..."
        rm -rf "$VECTORDB_DIR"/*
        rm -f "$VECTORDB_CHECK_FILE"
        mkdir -p "$VECTORDB_DIR"
        
        if test_vectordb_health; then
            touch "$VECTORDB_CHECK_FILE"
            echo "✅ Vector database reinitialized successfully!"
        else
            echo "❌ Vector database reinitialization failed!"
            exit 1
        fi
    fi
fi

# Setup documents directory
echo "Checking documents directory setup..."
if [ ! -d "$DOCUMENTS_DIR" ]; then
    echo "Creating documents directory..."
    mkdir -p "$DOCUMENTS_DIR"
    echo "✅ Documents directory created successfully!"
    echo "📄 You can now place your source documents in: $DOCUMENTS_DIR"
else
    echo "✅ Documents directory already exists, skipping setup."
fi

# Setup configuration file
echo "Checking configuration file setup..."
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Creating config.json file..."
    echo "{}" > "$CONFIG_FILE"
    echo "✅ Configuration file created successfully!"
    echo "⚙️  You can now customize settings in: $CONFIG_FILE"
else
    echo "✅ Configuration file already exists, skipping setup."
fi

# Check if server.py exists
if [ ! -f "$PYTHON_FILE" ]; then
    echo "Error: server.py not found in $SCRIPT_DIR"
    exit 1
fi

# Display optimization status before starting server
echo "Optimization Status:"
echo "==================="
python -c "
import os, platform
try:
    import llama_cpp
    print(f'llama-cpp-python: v{llama_cpp.__version__}')
    
    # Simple platform-based GPU detection
    if platform.system() == 'Darwin':
        print('GPU Acceleration: Metal (Apple GPU)')
    else:
        print('GPU Acceleration: Platform dependent')
        
    print(f'CPU Cores: {os.cpu_count()}')
    print(f'Platform: {platform.system()} {platform.machine()}')
    
except ImportError:
    print('llama-cpp-python: Not available')
except Exception as e:
    print(f'Status check error: {e}')
" 2>/dev/null || echo "Status check failed"

echo "================================"

# Run the MCP server
echo "Starting RAG MCP Server..."
echo "Press Ctrl+C to stop the server"
echo "================================"
python "$PYTHON_FILE"