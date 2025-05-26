#!/usr/bin/env python3
"""
Python 3.12 Uyumlu Kurulum Scripti
Trading Bot için Python 3.12 uyumlu paket kurulumu
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, check=True):
    """Komut çalıştır"""
    print(f"Running: {command}")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        raise

def check_python_version():
    """Python versiyonunu kontrol et"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 8:
        print("ERROR: Python 3.8+ required")
        sys.exit(1)
    
    if version.minor == 12:
        print("✅ Python 3.12 detected - using compatible packages")
        return True
    else:
        print(f"⚠️  Python {version.major}.{version.minor} detected - may need adjustments")
        return False

def install_system_dependencies():
    """Sistem bağımlılıklarını yükle"""
    system = platform.system().lower()
    
    if system == "linux":
        print("Installing Linux system dependencies...")
        try:
            # Ubuntu/Debian
            run_command("sudo apt update")
            run_command("sudo apt install -y python3-dev python3-pip build-essential libssl-dev libffi-dev")
        except:
            try:
                # CentOS/RHEL
                run_command("sudo yum install -y python3-devel python3-pip gcc openssl-devel libffi-devel")
            except:
                print("Could not install system dependencies automatically")
    
    elif system == "darwin":  # macOS
        print("Installing macOS system dependencies...")
        try:
            run_command("xcode-select --install", check=False)
        except:
            print("Xcode command line tools may already be installed")
    
    elif system == "windows":
        print("Windows detected - ensure Visual Studio Build Tools are installed")

def install_python_packages():
    """Python paketlerini yükle"""
    print("Installing Python packages...")
    
    # Pip'i güncelle
    run_command(f"{sys.executable} -m pip install --upgrade pip")
    
    # Setuptools ve wheel'i güncelle (Python 3.12 için önemli)
    run_command(f"{sys.executable} -m pip install --upgrade setuptools>=69.0.0 wheel>=0.42.0")
    
    # Core packages önce
    core_packages = [
        "numpy>=1.26.0",
        "pandas>=2.1.0", 
        "setuptools>=69.0.0",
        "wheel>=0.42.0"
    ]
    
    for package in core_packages:
        print(f"Installing {package}...")
        run_command(f"{sys.executable} -m pip install {package}")
    
    # Ana requirements.txt'i yükle
    if os.path.exists("requirements.txt"):
        print("Installing from requirements.txt...")
        run_command(f"{sys.executable} -m pip install -r requirements.txt")
    else:
        print("requirements.txt not found, installing individual packages...")
        install_individual_packages()

def install_individual_packages():
    """Paketleri tek tek yükle"""
    packages = [
        # Core
        "python-telegram-bot==20.7",
        "python-binance==1.0.19",
        "aiohttp>=3.9.0",
        
        # Data Science
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.4.0",
        
        # Technical Analysis
        "pandas-ta>=0.3.14b0",
        "yfinance>=0.2.18",
        
        # Database
        "sqlalchemy>=2.0.0",
        "aiosqlite>=0.19.0",
        
        # Async
        "asyncio-mqtt>=0.16.0",
        
        # Config
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        
        # Logging
        "structlog>=23.0.0",
        "colorlog>=6.8.0",
        
        # Security
        "cryptography>=42.0.0",
        "bcrypt>=4.1.0",
        
        # HTTP
        "requests>=2.31.0",
        
        # Utils
        "python-dateutil>=2.8.0",
        "pytz>=2023.3",
        "click>=8.1.0",
        
        # Testing
        "pytest>=7.4.0",
        "pytest-asyncio>=0.23.0",
        
        # Performance
        "psutil>=5.9.0",
        
        # ML (optional)
        "tensorflow>=2.15.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.3.0",
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            run_command(f"{sys.executable} -m pip install {package}")
        except Exception as e:
            print(f"Failed to install {package}: {e}")
            print("Continuing with other packages...")

def create_virtual_environment():
    """Virtual environment oluştur"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("Virtual environment already exists")
        return
    
    print("Creating virtual environment...")
    run_command(f"{sys.executable} -m venv venv")
    
    # Activation script'ini göster
    system = platform.system().lower()
    if system == "windows":
        activate_script = "venv\\Scripts\\activate.bat"
        print(f"To activate: {activate_script}")
    else:
        activate_script = "source venv/bin/activate"
        print(f"To activate: {activate_script}")

def test_installation():
    """Kurulumu test et"""
    print("\nTesting installation...")
    
    test_imports = [
        "pandas",
        "numpy", 
        "telegram",
        "binance",
        "aiohttp",
        "sqlalchemy",
        "pandas_ta",
        "cryptography",
        "pydantic"
    ]
    
    failed_imports = []
    
    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {failed_imports}")
        print("You may need to install these manually:")
        for module in failed_imports:
            print(f"  pip install {module}")
    else:
        print("\n✅ All imports successful!")

def main():
    """Ana fonksiyon"""
    print("🤖 Trading Bot - Python 3.12 Compatible Installation")
    print("=" * 50)
    
    # Python version check
    is_python312 = check_python_version()
    
    # System dependencies
    try:
        install_system_dependencies()
    except Exception as e:
        print(f"Warning: Could not install system dependencies: {e}")
    
    # Virtual environment (opsiyonel)
    create_venv = input("\nCreate virtual environment? (y/n): ").lower().strip()
    if create_venv in ['y', 'yes']:
        create_virtual_environment()
        print("\nPlease activate the virtual environment and run this script again.")
        return
    
    # Python packages
    try:
        install_python_packages()
    except Exception as e:
        print(f"Error installing packages: {e}")
        print("Trying individual package installation...")
        install_individual_packages()
    
    # Test
    test_installation()
    
    print("\n" + "=" * 50)
    print("🎉 Installation completed!")
    print("\nNext steps:")
    print("1. Copy .env.example to .env")
    print("2. Edit .env with your API keys")
    print("3. Run: python main.py")

if __name__ == "__main__":
    main()
