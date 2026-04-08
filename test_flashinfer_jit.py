#!/usr/bin/env python3
"""
Diagnostic script for FlashInfer GDN prefill JIT compile hang issue.

Run this script on the Linux server where vLLM is running:
    python test_flashinfer_jit.py

Or copy to remote server and run:
    scp test_flashinfer_jit.py user@server:/tmp/
    ssh user@server "python /tmp/test_flashinfer_jit.py"
"""

# IMPORTANT: Set these BEFORE importing flashinfer to limit compilation concurrency
import os
os.environ['MAX_JOBS'] = os.environ.get('MAX_JOBS', '2')
os.environ['NVCC_THREADS'] = os.environ.get('NVCC_THREADS', '1')

import sys
import time
import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('flashinfer_diagnostic')

def check_environment():
    """Check build environment."""
    logger.info("=== Environment Check ===")
    
    # Check ninja
    try:
        result = subprocess.run(['ninja', '--version'], capture_output=True, text=True, timeout=10)
        logger.info(f"ninja version: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("ninja NOT FOUND - this is required for FlashInfer JIT")
    except subprocess.TimeoutExpired:
        logger.error("ninja command timed out")
    except Exception as e:
        logger.error(f"ninja check failed: {e}")
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
        logger.info(f"nvcc version: {result.stdout.strip()}")
    except FileNotFoundError:
        logger.error("nvcc NOT FOUND - CUDA compiler required for FlashInfer JIT")
    except Exception as e:
        logger.error(f"nvcc check failed: {e}")
    
    # Check g++
    try:
        result = subprocess.run(['g++', '--version'], capture_output=True, text=True, timeout=10)
        logger.info(f"g++ version: {result.stdout.split(chr(10))[0]}")
    except Exception as e:
        logger.error(f"g++ check failed: {e}")
    
    # Environment variables
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    logger.info(f"FLASHINFER_CACHE_DIR: {os.environ.get('FLASHINFER_CACHE_DIR', 'not set')}")
    logger.info(f"MAX_JOBS: {os.environ.get('MAX_JOBS', 'not set')}")
    logger.info(f"NVCC_THREADS: {os.environ.get('NVCC_THREADS', 'not set')}")
    
    # Python info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Python executable: {sys.executable}")

def check_flashinfer():
    """Check FlashInfer installation."""
    logger.info("=== FlashInfer Check ===")
    
    try:
        import flashinfer
        logger.info(f"FlashInfer version: {flashinfer.__version__}")
        logger.info(f"FlashInfer location: {flashinfer.__file__}")
        
        # Check if prebuilt cubins are available
        try:
            from flashinfer.utils import has_prebuilt_cubin
            has_cubin = has_prebuilt_cubin()
            logger.info(f"Has prebuilt cubin: {has_cubin}")
        except Exception as e:
            logger.warning(f"Could not check prebuilt cubin: {e}")
        
    except ImportError as e:
        logger.error(f"FlashInfer import failed: {e}")
        return False
    
    return True

def test_jit_compile():
    """Test FlashInfer JIT compilation with limited concurrency."""
    logger.info("=== JIT Compile Test ===")
    logger.info(f"MAX_JOBS: {os.environ['MAX_JOBS']}")
    logger.info(f"NVCC_THREADS: {os.environ['NVCC_THREADS']}")
    
    try:
        logger.info("Importing flashinfer.gdn_prefill (this may trigger JIT)...")
        start = time.time()
        
        # This will trigger JIT compilation if not cached
        from flashinfer.gdn_prefill import get_gdn_prefill_module
        
        logger.info("Getting GDN prefill module...")
        
        # Use a timeout to detect hangs
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("JIT compilation timed out after 60 seconds")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)  # 60 second timeout
        
        try:
            module = get_gdn_prefill_module()
            signal.alarm(0)  # Cancel alarm
            
            logger.info(f"Module obtained in {time.time() - start:.2f} seconds")
            logger.info(f"Module: {module}")
            logger.info(f"Module build_dir: {getattr(module, 'build_dir', 'unknown')}")
            
            # Check if module is already compiled
            if hasattr(module, 'built'):
                logger.info(f"Module already built: {module.built}")
            
            # Try to build and load
            logger.info("Building and loading module...")
            start_build = time.time()
            
            # Set longer timeout for build
            signal.alarm(120)  # 2 minute timeout for build
            
            # Add verbose logging
            logger.info("Calling build_and_load() with verbose=True...")
            loaded = module.build_and_load(verbose=True)
            signal.alarm(0)  # Cancel alarm
            
            logger.info(f"Module built and loaded in {time.time() - start_build:.2f} seconds")
            logger.info(f"Loaded module: {loaded}")
            
            return True
            
        except TimeoutError as e:
            logger.error(str(e))
            logger.error("JIT compilation is hanging - this is the problem!")
            
            # Try to find what's running
            logger.info("=== Checking running processes ===")
            try:
                import subprocess
                result = subprocess.run(
                    ['ps', 'aux'],
                    capture_output=True, text=True, timeout=10
                )
                # Filter for relevant processes
                for line in result.stdout.split('\n'):
                    if any(x in line for x in ['ninja', 'nvcc', 'g++', 'ptxas', 'cudafe', 'cicc']):
                        logger.info(f"Running: {line}")
            except Exception as e:
                logger.warning(f"Could not check processes: {e}")
            
            return False
        except Exception as e:
            logger.error(f"JIT compile failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_cache_dir():
    """Check FlashInfer cache directory."""
    logger.info("=== Cache Directory Check ===")
    
    # Default cache locations
    possible_cache_dirs = [
        os.environ.get('FLASHINFER_CACHE_DIR'),
        os.path.expanduser('~/.cache/flashinfer'),
        os.path.join(os.path.dirname(__file__), '.flashinfer_cache'),
    ]
    
    for cache_dir in possible_cache_dirs:
        if cache_dir:
            if os.path.exists(cache_dir):
                logger.info(f"Cache dir exists: {cache_dir}")
                # List contents
                try:
                    files = os.listdir(cache_dir)
                    logger.info(f"  Contents ({len(files)} items): {files[:10]}{'...' if len(files) > 10 else ''}")
                except Exception as e:
                    logger.warning(f"  Could not list contents: {e}")
            else:
                logger.info(f"Cache dir does not exist: {cache_dir}")

def main():
    logger.info("Starting FlashInfer GDN Prefill JIT Diagnostic")
    logger.info("=" * 60)
    
    check_environment()
    logger.info("")
    
    check_cache_dir()
    logger.info("")
    
    if not check_flashinfer():
        logger.error("FlashInfer not available, stopping.")
        return 1
    
    logger.info("")
    
    success = test_jit_compile()
    
    logger.info("")
    logger.info("=" * 60)
    if success:
        logger.info("DIAGNOSTIC PASSED - FlashInfer JIT working correctly")
    else:
        logger.info("DIAGNOSTIC FAILED - FlashInfer JIT has issues")
        logger.info("")
        logger.info("Recommendations:")
        logger.info("1. Use --gdn-prefill-backend triton to avoid JIT")
        logger.info("2. Check ninja/nvcc installation")
        logger.info("3. Set FLASHINFER_CACHE_DIR to persist compiled modules")
        logger.info("4. Try: pip install flashinfer-jit-cache")
    
    return 0 if success else 1

if __name__ == '__main__':
    sys.exit(main())
