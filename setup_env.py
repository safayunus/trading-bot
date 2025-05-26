#!/usr/bin/env python3
"""
Environment Setup Script
.env dosyası oluşturma ve API anahtarları ayarlama rehberi
"""

import os
import sys
from pathlib import Path

def create_env_file():
    """
    .env dosyası oluştur ve kullanıcıdan API anahtarlarını al
    """
    print("🔧 Trading Bot Environment Setup")
    print("=" * 50)
    
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    # .env.example'dan başla
    if env_example.exists():
        print("✅ .env.example dosyası bulundu")
        with open(env_example, 'r') as f:
            env_content = f.read()
    else:
        print("⚠️  .env.example bulunamadı, varsayılan template oluşturuluyor...")
        env_content = create_default_env_template()
    
    print("\n📝 API Anahtarlarınızı girmeniz gerekiyor:")
    print("(Şimdi girmek istemiyorsanız ENTER'a basın, sonra manuel düzenleyebilirsiniz)")
    
    # Telegram Bot Token
    print("\n🤖 TELEGRAM BOT TOKEN:")
    print("1. Telegram'da @BotFather'a mesaj gönderin")
    print("2. /newbot komutunu kullanın")
    print("3. Bot adını ve kullanıcı adını belirleyin")
    print("4. Aldığınız token'ı aşağıya girin")
    
    telegram_token = input("\nTelegram Bot Token (opsiyonel): ").strip()
    
    # Telegram Chat ID
    print("\n💬 TELEGRAM CHAT ID:")
    print("1. Bot'unuza bir mesaj gönderin")
    print("2. Bu URL'yi ziyaret edin: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates")
    print("3. 'chat.id' değerini bulun")
    
    telegram_chat_id = input("Telegram Chat ID (opsiyonel): ").strip()
    
    # Binance API
    print("\n💰 BINANCE API ANAHTARLARI:")
    print("1. Binance hesabınıza giriş yapın")
    print("2. Account > API Management'a gidin")
    print("3. 'Create API' butonuna tıklayın")
    print("4. API Key ve Secret Key'i kopyalayın")
    print("⚠️  UYARI: Withdraw izinlerini kapatın!")
    
    binance_api_key = input("Binance API Key (opsiyonel): ").strip()
    binance_secret_key = input("Binance Secret Key (opsiyonel): ").strip()
    
    # Test mode
    print("\n🧪 TEST MODE:")
    print("Başlangıçta testnet kullanmanız önerilir")
    use_testnet = input("Testnet kullanmak istiyor musunuz? (y/n) [y]: ").strip().lower()
    if use_testnet in ['', 'y', 'yes']:
        testnet = "true"
    else:
        testnet = "false"
    
    # Environment değişkenlerini güncelle
    updated_content = env_content
    
    if telegram_token:
        updated_content = update_env_var(updated_content, "TELEGRAM_BOT_TOKEN", telegram_token)
        updated_content = update_env_var(updated_content, "TELEGRAM_TOKEN", telegram_token)
    
    if telegram_chat_id:
        updated_content = update_env_var(updated_content, "TELEGRAM_CHAT_ID", telegram_chat_id)
    
    if binance_api_key:
        updated_content = update_env_var(updated_content, "BINANCE_API_KEY", binance_api_key)
    
    if binance_secret_key:
        updated_content = update_env_var(updated_content, "BINANCE_SECRET_KEY", binance_secret_key)
    
    updated_content = update_env_var(updated_content, "BINANCE_TESTNET", testnet)
    
    # .env dosyasını kaydet
    with open(env_file, 'w') as f:
        f.write(updated_content)
    
    print(f"\n✅ .env dosyası oluşturuldu: {env_file.absolute()}")
    
    # Eksik anahtarları kontrol et
    missing_keys = []
    if not telegram_token:
        missing_keys.append("TELEGRAM_BOT_TOKEN")
    if not telegram_chat_id:
        missing_keys.append("TELEGRAM_CHAT_ID")
    if not binance_api_key:
        missing_keys.append("BINANCE_API_KEY")
    if not binance_secret_key:
        missing_keys.append("BINANCE_SECRET_KEY")
    
    if missing_keys:
        print(f"\n⚠️  Eksik API anahtarları: {', '.join(missing_keys)}")
        print(f"Bu anahtarları daha sonra .env dosyasını düzenleyerek ekleyebilirsiniz:")
        print(f"nano .env  # Linux/macOS")
        print(f"notepad .env  # Windows")
    
    return len(missing_keys) == 0

def create_default_env_template():
    """Varsayılan .env template oluştur"""
    return """# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ADMIN_ID=your_admin_id_here

# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_TESTNET=true

# Database Configuration
DATABASE_URL=sqlite:///trading_bot.db

# Security
SECRET_KEY=your_very_secure_secret_key_here
ENCRYPTION_KEY=your_32_character_encryption_key

# Trading Configuration
DEFAULT_TRADE_AMOUNT=100
MAX_OPEN_POSITIONS=5
RISK_PERCENTAGE=2.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log

# Performance
ASYNC_WORKERS=4
MAX_CONCURRENT_REQUESTS=10
"""

def update_env_var(content, var_name, value):
    """Environment değişkenini güncelle"""
    lines = content.split('\n')
    updated = False
    
    for i, line in enumerate(lines):
        if line.startswith(f"{var_name}="):
            lines[i] = f"{var_name}={value}"
            updated = True
            break
    
    if not updated:
        lines.append(f"{var_name}={value}")
    
    return '\n'.join(lines)

def test_configuration():
    """Konfigürasyonu test et"""
    print("\n🧪 Konfigürasyon testi...")
    
    try:
        # Config modülünü import etmeyi dene
        sys.path.insert(0, '.')
        from config import Config
        
        config = Config()
        print("✅ Konfigürasyon başarıyla yüklendi")
        
        # Temel ayarları kontrol et
        checks = [
            ("Telegram Token", hasattr(config, 'TELEGRAM_TOKEN') and config.TELEGRAM_TOKEN != "your_bot_token_here"),
            ("Binance API Key", hasattr(config, 'BINANCE_API_KEY') and config.BINANCE_API_KEY != "your_binance_api_key"),
            ("Database URL", hasattr(config, 'DATABASE_URL')),
            ("Log Level", hasattr(config, 'LOG_LEVEL')),
        ]
        
        for check_name, check_result in checks:
            if check_result:
                print(f"✅ {check_name}")
            else:
                print(f"⚠️  {check_name} - Eksik veya varsayılan değer")
        
        return True
        
    except Exception as e:
        print(f"❌ Konfigürasyon hatası: {e}")
        return False

def create_demo_mode():
    """Demo mode için minimal konfigürasyon"""
    print("\n🎮 DEMO MODE KURULUMU")
    print("API anahtarları olmadan bot'u test edebilirsiniz")
    
    demo_env = """# Demo Mode Configuration
TELEGRAM_BOT_TOKEN=demo_mode
TELEGRAM_TOKEN=demo_mode
TELEGRAM_CHAT_ID=123456789
TELEGRAM_ADMIN_ID=123456789

# Demo Binance (Testnet)
BINANCE_API_KEY=demo_mode
BINANCE_SECRET_KEY=demo_mode
BINANCE_TESTNET=true

# Database
DATABASE_URL=sqlite:///trading_bot_demo.db

# Security
SECRET_KEY=demo_secret_key_for_testing_only
ENCRYPTION_KEY=demo_encryption_key_32_characters

# Trading (Demo)
DEFAULT_TRADE_AMOUNT=10
MAX_OPEN_POSITIONS=3
RISK_PERCENTAGE=1.0

# Logging
LOG_LEVEL=DEBUG
LOG_FILE=logs/trading_bot_demo.log

# Demo Mode Flag
DEMO_MODE=true
"""
    
    with open('.env', 'w') as f:
        f.write(demo_env)
    
    print("✅ Demo mode .env dosyası oluşturuldu")
    print("⚠️  Bu sadece test amaçlıdır, gerçek trading yapmaz")

def main():
    """Ana fonksiyon"""
    print("🤖 Trading Bot Environment Setup")
    print("=" * 50)
    
    # Mevcut .env dosyasını kontrol et
    if Path(".env").exists():
        print("⚠️  .env dosyası zaten mevcut")
        overwrite = input("Üzerine yazmak istiyor musunuz? (y/n): ").strip().lower()
        if overwrite not in ['y', 'yes']:
            print("İşlem iptal edildi")
            return
    
    print("\nKurulum seçenekleri:")
    print("1. Tam kurulum (API anahtarları ile)")
    print("2. Demo mode (API anahtarları olmadan test)")
    print("3. Manuel düzenleme için boş template")
    
    choice = input("\nSeçiminiz (1-3) [1]: ").strip()
    
    if choice == "2":
        create_demo_mode()
    elif choice == "3":
        # Sadece template oluştur
        env_content = create_default_env_template()
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Boş .env template oluşturuldu")
        print("Lütfen .env dosyasını düzenleyerek API anahtarlarınızı ekleyin")
    else:
        # Tam kurulum
        all_configured = create_env_file()
        
        if all_configured:
            # Konfigürasyonu test et
            if test_configuration():
                print("\n🎉 Kurulum tamamlandı!")
                print("Bot'u başlatmak için: python main.py")
            else:
                print("\n⚠️  Konfigürasyon testi başarısız")
                print("Lütfen .env dosyasını kontrol edin")
        else:
            print("\n⚠️  Bazı API anahtarları eksik")
            print("Bot'u çalıştırmadan önce .env dosyasını tamamlayın")
    
    print(f"\n📁 .env dosyası konumu: {Path('.env').absolute()}")
    print("📝 Düzenlemek için: nano .env (Linux/macOS) veya notepad .env (Windows)")

if __name__ == "__main__":
    main()
