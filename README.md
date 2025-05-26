# 🤖 AI-Powered Cryptocurrency Trading Bot

**Telegram kontrolünde çalışan, yapay zeka destekli cryptocurrency trading botu**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)]()

## 📋 İçindekiler

- [🎯 Özellikler](#-özellikler)
- [🏗️ Sistem Gereksinimleri](#️-sistem-gereksinimleri)
- [⚡ Hızlı Başlangıç](#-hızlı-başlangıç)
- [🔧 Kurulum](#-kurulum)
- [⚙️ Konfigürasyon](#️-konfigürasyon)
- [🚀 Kullanım](#-kullanım)
- [📊 Monitoring](#-monitoring)
- [🧪 Test](#-test)
- [🔒 Güvenlik](#-güvenlik)
- [🚀 Production Deployment](#-production-deployment)
- [🔧 Troubleshooting](#-troubleshooting)
- [📈 Performance Tuning](#-performance-tuning)
- [🤝 Katkıda Bulunma](#-katkıda-bulunma)

## 🎯 Özellikler

### 🤖 **Yapay Zeka & Machine Learning**
- **LSTM Neural Networks** - Fiyat tahmin modeli
- **Technical Analysis** - 20+ teknik indikatör
- **Sentiment Analysis** - Piyasa duygu analizi
- **Ensemble Models** - Çoklu model kombinasyonu
- **Real-time Learning** - Sürekli öğrenme sistemi

### 📱 **Telegram Bot Kontrolü**
- **Komut Tabanlı Kontrol** - `/start`, `/status`, `/trade` komutları
- **Real-time Bildirimler** - Anlık trade bildirimleri
- **Interactive Buttons** - Kolay kullanım arayüzü
- **Multi-user Support** - Çoklu kullanıcı desteği
- **Admin Panel** - Yönetici kontrol paneli

### 💹 **Trading Özellikleri**
- **Binance API Entegrasyonu** - Spot ve futures trading
- **Multiple Strategies** - Çoklu trading stratejileri
- **Risk Management** - Gelişmiş risk yönetimi
- **Position Management** - Otomatik pozisyon yönetimi
- **Stop Loss & Take Profit** - Otomatik kar/zarar durdurma

### 🛡️ **Risk Yönetimi**
- **Portfolio Risk Control** - Portföy risk kontrolü
- **Position Sizing** - Dinamik pozisyon boyutlandırma
- **Drawdown Protection** - Kayıp koruma sistemi
- **Emergency Stop** - Acil durdurma mekanizması
- **VaR Calculation** - Value at Risk hesaplama

### 📊 **Monitoring & Analytics**
- **Real-time Dashboard** - Canlı performans takibi
- **Performance Metrics** - Detaylı performans metrikleri
- **Trade History** - Kapsamlı trade geçmişi
- **System Health** - Sistem sağlık kontrolü
- **Memory Management** - Otomatik memory optimizasyonu

### 🔒 **Güvenlik**
- **API Key Encryption** - API anahtarı şifreleme
- **Rate Limiting** - İstek sınırlama
- **Input Validation** - Girdi doğrulama
- **Secure Configuration** - Güvenli konfigürasyon
- **Audit Logging** - Güvenlik log kayıtları

## 🏗️ Sistem Gereksinimleri

### **Minimum Gereksinimler**
- **İşletim Sistemi:** Windows 10/11, Ubuntu 18.04+, macOS 10.15+
- **Python:** 3.8 veya üzeri
- **RAM:** 4 GB (8 GB önerilen)
- **Disk:** 2 GB boş alan
- **İnternet:** Stabil internet bağlantısı

### **Önerilen Gereksinimler**
- **İşletim Sistemi:** Ubuntu 20.04 LTS (Production için)
- **Python:** 3.9+
- **RAM:** 8 GB veya üzeri
- **Disk:** 10 GB SSD
- **CPU:** 4 core veya üzeri
- **İnternet:** Fiber internet bağlantısı

### **Gerekli Hesaplar**
- **Binance Hesabı** - API anahtarları ile
- **Telegram Bot Token** - @BotFather'dan alınacak
- **Telegram Chat ID** - Bot ile iletişim için

## ⚡ Hızlı Başlangıç

### **1. Repository'yi İndirin**
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

### **2. Sanal Ortam Oluşturun**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### **3. Bağımlılıkları Yükleyin**
```bash
pip install -r requirements.txt
```

### **4. Konfigürasyonu Ayarlayın**
```bash
# .env dosyasını oluşturun
cp .env.example .env

# .env dosyasını düzenleyin
notepad .env  # Windows
nano .env     # Linux
```

### **5. Botu Başlatın**
```bash
python main.py
```

## 🔧 Kurulum

### **Detaylı Kurulum Adımları**

#### **1. Python Kurulumu**

**Windows:**
1. [Python.org](https://python.org)'dan Python 3.9+ indirin
2. Kurulum sırasında "Add Python to PATH" seçeneğini işaretleyin
3. Kurulumu tamamlayın

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**macOS:**
```bash
# Homebrew ile
brew install python3

# Veya Python.org'dan indirin
```

#### **2. Git Kurulumu**

**Windows:**
- [Git for Windows](https://git-scm.com/download/win) indirin ve kurun

**Ubuntu/Debian:**
```bash
sudo apt install git
```

**macOS:**
```bash
brew install git
```

#### **3. Repository Klonlama**
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

#### **4. Sanal Ortam Kurulumu**
```bash
# Sanal ortam oluştur
python -m venv venv

# Sanal ortamı aktifleştir
# Windows:
venv\Scripts\activate

# Linux/macOS:
source venv/bin/activate

# Pip'i güncelleyin
pip install --upgrade pip
```

#### **5. Bağımlılık Kurulumu**
```bash
# Tüm bağımlılıkları yükle
pip install -r requirements.txt

# TA-Lib kurulumu (Windows için özel)
# Windows'ta TA-Lib kurulumu zor olabilir
# Alternatif: talib-binary kullanın (requirements.txt'te mevcut)
```

#### **6. Database Kurulumu**
```bash
# SQLite database otomatik oluşturulacak
# Manuel oluşturmak için:
python -c "from utils.database import init_database; init_database()"
```

## ⚙️ Konfigürasyon

### **1. Environment Variables (.env)**

`.env.example` dosyasını `.env` olarak kopyalayın ve düzenleyin:

```bash
cp .env.example .env
```

**Gerekli Ayarlar:**
```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ADMIN_ID=your_admin_id_here

# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_SECRET_KEY=your_binance_secret_key
BINANCE_TESTNET=true  # Production için false yapın

# Database Configuration
DATABASE_URL=sqlite:///trading_bot.db

# Security
SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# Trading Configuration
DEFAULT_TRADE_AMOUNT=100
MAX_OPEN_POSITIONS=5
RISK_PERCENTAGE=2.0

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log
```

### **2. Telegram Bot Kurulumu**

#### **Bot Token Alma:**
1. Telegram'da @BotFather'a mesaj gönderin
2. `/newbot` komutunu kullanın
3. Bot adını ve kullanıcı adını belirleyin
4. Aldığınız token'ı `.env` dosyasına ekleyin

#### **Chat ID Alma:**
1. Bot'unuza mesaj gönderin
2. Bu URL'yi ziyaret edin: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. `chat.id` değerini bulun ve `.env` dosyasına ekleyin

### **3. Binance API Kurulumu**

#### **API Anahtarları Alma:**
1. [Binance](https://binance.com) hesabınıza giriş yapın
2. Account > API Management'a gidin
3. "Create API" butonuna tıklayın
4. API Key ve Secret Key'i kopyalayın
5. IP kısıtlaması ekleyin (güvenlik için)

#### **API İzinleri:**
- ✅ **Spot & Margin Trading** - Spot trading için
- ✅ **Futures** - Futures trading için (opsiyonel)
- ❌ **Withdraw** - Güvenlik için kapatın

### **4. Risk Ayarları**

`config.py` dosyasında risk parametrelerini ayarlayın:

```python
# Risk Management
RISK_SETTINGS = {
    'max_risk_per_trade': 2.0,      # Trade başına max risk (%)
    'max_portfolio_risk': 10.0,     # Portföy max risk (%)
    'max_drawdown': 15.0,           # Max drawdown (%)
    'position_size_method': 'kelly', # kelly, fixed, percent
    'stop_loss_percentage': 3.0,    # Stop loss (%)
    'take_profit_percentage': 6.0,  # Take profit (%)
}
```

## 🚀 Kullanım

### **1. Bot'u Başlatma**

#### **Development Mode:**
```bash
# Sanal ortamı aktifleştir
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Bot'u başlat
python main.py
```

#### **Production Mode:**
```bash
# Systemd service olarak çalıştır (Linux)
sudo systemctl start trading-bot
sudo systemctl enable trading-bot

# Veya screen/tmux ile
screen -S trading-bot python main.py
```

### **2. Telegram Komutları**

#### **Temel Komutlar:**
```
/start          - Bot'u başlat
/help           - Yardım menüsü
/status         - Bot durumu
/balance        - Hesap bakiyesi
/positions      - Açık pozisyonlar
/history        - Trade geçmişi
/stop           - Bot'u durdur
```

#### **Trading Komutları:**
```
/trade BTCUSDT buy 100    - Manuel trade
/close BTCUSDT            - Pozisyon kapat
/strategy list            - Strateji listesi
/strategy start ma_cross  - Strateji başlat
/strategy stop ma_cross   - Strateji durdur
```

#### **Monitoring Komutları:**
```
/performance    - Performans raporu
/risk          - Risk analizi
/health        - Sistem sağlığı
/logs          - Son loglar
/backup        - Database backup
```

#### **Admin Komutları:**
```
/admin users           - Kullanıcı listesi
/admin settings        - Bot ayarları
/admin restart         - Bot'u yeniden başlat
/admin emergency_stop  - Acil durdurma
```

### **3. Web Dashboard (Opsiyonel)**

```bash
# Flask dashboard başlat
python -m flask run --host=0.0.0.0 --port=5000

# Dashboard'a erişim
http://localhost:5000
```

## 📊 Monitoring

### **1. Real-time Monitoring**

#### **Telegram Bildirimleri:**
- 📈 **Trade Bildirimleri** - Her trade için otomatik bildirim
- ⚠️ **Risk Uyarıları** - Risk eşikleri aşıldığında
- 🔴 **Hata Bildirimleri** - Sistem hataları için
- 📊 **Günlük Raporlar** - Günlük performans özeti

#### **Log Monitoring:**
```bash
# Real-time log takibi
tail -f logs/trading_bot.log

# Error logları
grep ERROR logs/trading_bot.log

# Trade logları
grep TRADE logs/trading_bot.log
```

### **2. Performance Metrics**

#### **Key Performance Indicators (KPI):**
- **Total Return** - Toplam getiri
- **Sharpe Ratio** - Risk-adjusted return
- **Max Drawdown** - Maksimum kayıp
- **Win Rate** - Kazanma oranı
- **Profit Factor** - Kar faktörü
- **Average Trade** - Ortalama trade

#### **System Metrics:**
- **Memory Usage** - Memory kullanımı
- **CPU Usage** - CPU kullanımı
- **API Response Time** - API yanıt süresi
- **Trade Execution Time** - Trade gerçekleştirme süresi

### **3. Alerting System**

#### **Alert Türleri:**
- 🔴 **Critical** - Sistem kritik hatalar
- 🟡 **Warning** - Uyarı seviyesi
- 🔵 **Info** - Bilgi amaçlı
- 🟢 **Success** - Başarılı işlemler

## 🧪 Test

### **1. Unit Tests**

```bash
# Tüm testleri çalıştır
python tests/run_tests.py --all

# Belirli modül testleri
python tests/run_tests.py --files tests/test_telegram.py

# Coverage ile
python -m pytest --cov=. --cov-report=html
```

### **2. Integration Tests**

```bash
# Integration testleri
python tests/run_tests.py --integration

# API testleri (testnet gerekli)
python tests/run_tests.py --api
```

### **3. Backtesting**

```bash
# Backtest çalıştır
python tests/backtesting.py --strategy ma_cross --days 30

# Strateji karşılaştırması
python tests/backtesting.py --compare --strategies ma_cross,rsi_strategy
```

### **4. Paper Trading**

```bash
# Paper trading mode
python main.py --paper-trading

# Testnet trading
python main.py --testnet
```

## 🔒 Güvenlik

### **1. API Güvenliği**

#### **Best Practices:**
- ✅ **IP Whitelist** - API anahtarlarını IP ile sınırlayın
- ✅ **Read-Only Keys** - Mümkünse sadece okuma izni
- ✅ **Regular Rotation** - API anahtarlarını düzenli değiştirin
- ✅ **Environment Variables** - Anahtarları kod içinde saklamayın

#### **API İzinleri:**
```python
# Minimum gerekli izinler
REQUIRED_PERMISSIONS = [
    'spot_trading',      # Spot trading
    'margin_trading',    # Margin trading (opsiyonel)
    'futures_trading',   # Futures trading (opsiyonel)
]

# Güvenlik için kapatılması gerekenler
FORBIDDEN_PERMISSIONS = [
    'withdraw',          # Para çekme
    'internal_transfer', # İç transfer
    'sub_account',       # Alt hesap yönetimi
]
```

### **2. Sistem Güvenliği**

#### **Firewall Ayarları:**
```bash
# Ubuntu UFW
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 5000/tcp  # Dashboard (opsiyonel)
sudo ufw deny 22/tcp from any to any  # SSH'ı sınırla
```

#### **SSL/TLS:**
```bash
# Let's Encrypt ile SSL
sudo apt install certbot
sudo certbot --nginx -d yourdomain.com
```

### **3. Data Protection**

#### **Encryption:**
- 🔐 **API Keys** - AES-256 ile şifrelenmiş
- 🔐 **Database** - Hassas veriler şifrelenmiş
- 🔐 **Logs** - Kişisel bilgiler maskelenmiş
- 🔐 **Backups** - Şifrelenmiş backup'lar

#### **Access Control:**
```python
# Kullanıcı yetkileri
USER_ROLES = {
    'admin': ['all_commands'],
    'trader': ['trading_commands', 'view_commands'],
    'viewer': ['view_commands'],
}
```

## 🚀 Production Deployment

### **1. Server Kurulumu**

#### **Ubuntu Server Hazırlığı:**
```bash
# Sistem güncellemesi
sudo apt update && sudo apt upgrade -y

# Gerekli paketler
sudo apt install -y python3 python3-pip python3-venv git nginx supervisor

# Kullanıcı oluştur
sudo adduser tradingbot
sudo usermod -aG sudo tradingbot
```

#### **Application Deployment:**
```bash
# Application dizini
sudo mkdir -p /opt/trading-bot
sudo chown tradingbot:tradingbot /opt/trading-bot

# Repository clone
cd /opt/trading-bot
git clone https://github.com/yourusername/trading-bot.git .

# Virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **2. Systemd Service**

#### **Service Dosyası:**
```bash
# /etc/systemd/system/trading-bot.service
sudo cp deployment/trading-bot.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
```

#### **Service Kontrolü:**
```bash
# Status kontrol
sudo systemctl status trading-bot

# Logları görüntüle
sudo journalctl -u trading-bot -f

# Restart
sudo systemctl restart trading-bot
```

### **3. Nginx Reverse Proxy**

#### **Nginx Konfigürasyonu:**
```nginx
# /etc/nginx/sites-available/trading-bot
server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### **4. SSL Kurulumu**

```bash
# Certbot kurulumu
sudo apt install certbot python3-certbot-nginx

# SSL sertifikası
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# 0 12 * * * /usr/bin/certbot renew --quiet
```

### **5. Monitoring & Backup**

#### **Automated Backup:**
```bash
# Crontab backup
0 2 * * * /opt/trading-bot/deployment/backup.sh

# Backup script
chmod +x deployment/backup.sh
```

#### **Log Rotation:**
```bash
# /etc/logrotate.d/trading-bot
/opt/trading-bot/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 644 tradingbot tradingbot
}
```

## 🔧 Troubleshooting

### **1. Yaygın Sorunlar**

#### **Bot Başlamıyor:**
```bash
# Log kontrolü
tail -f logs/trading_bot.log

# Python path kontrolü
which python
python --version

# Bağımlılık kontrolü
pip list | grep -E "(telegram|binance|pandas)"
```

#### **API Bağlantı Hatası:**
```bash
# Network testi
ping api.binance.com

# API key testi
python -c "
from binance.client import Client
client = Client('your_api_key', 'your_secret')
print(client.get_account())
"
```

#### **Telegram Bot Yanıt Vermiyor:**
```bash
# Bot token testi
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getMe"

# Webhook kontrolü
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getWebhookInfo"
```

### **2. Performance Sorunları**

#### **Yüksek Memory Kullanımı:**
```bash
# Memory monitoring
python -c "
from utils.memory_manager import memory_manager
print(memory_manager.get_memory_report())
"

# Memory cleanup
python -c "
from utils.memory_manager import memory_manager
memory_manager.emergency_cleanup()
"
```

#### **Yavaş API Yanıtları:**
```bash
# API response time test
python tests/debugging.py --test api_performance

# Connection pool optimization
# config.py'da connection pool ayarları
```

### **3. Database Sorunları**

#### **Database Corruption:**
```bash
# SQLite integrity check
sqlite3 trading_bot.db "PRAGMA integrity_check;"

# Database backup restore
cp backups/trading_bot_backup_YYYYMMDD.db trading_bot.db
```

#### **Migration Issues:**
```bash
# Database reset (DİKKAT: Tüm veri silinir)
rm trading_bot.db
python -c "from utils.database import init_database; init_database()"
```

### **4. Debug Mode**

#### **Verbose Logging:**
```bash
# Debug mode ile başlat
LOG_LEVEL=DEBUG python main.py

# Specific module debug
python -c "
import logging
logging.getLogger('exchange.binance_client').setLevel(logging.DEBUG)
"
```

## 📈 Performance Tuning

### **1. System Optimization**

#### **Python Optimization:**
```bash
# PyPy kullanımı (opsiyonel)
pip install pypy3

# Cython compilation
python setup.py build_ext --inplace
```

#### **Memory Optimization:**
```python
# config.py'da memory ayarları
MEMORY_SETTINGS = {
    'warning_threshold': 80.0,
    'critical_threshold': 90.0,
    'cleanup_interval': 300,  # 5 dakika
    'max_cache_size': 1000,
}
```

### **2. Database Optimization**

#### **SQLite Tuning:**
```sql
-- SQLite performance ayarları
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = 10000;
PRAGMA temp_store = MEMORY;
```

#### **Index Optimization:**
```sql
-- Performans için indexler
CREATE INDEX idx_trades_timestamp ON trades(timestamp);
CREATE INDEX idx_trades_symbol ON trades(symbol);
CREATE INDEX idx_positions_status ON positions(status);
```

### **3. Network Optimization**

#### **Connection Pooling:**
```python
# Async connection pool ayarları
ASYNC_SETTINGS = {
    'max_connections': 100,
    'max_connections_per_host': 20,
    'timeout': 30,
    'keepalive_timeout': 30,
}
```

#### **Rate Limiting:**
```python
# API rate limiting
RATE_LIMITS = {
    'binance_spot': {'requests': 1200, 'window': 60},
    'binance_futures': {'requests': 2400, 'window': 60},
    'telegram': {'requests': 30, 'window': 1},
}
```

### **4. Caching Strategy**

#### **Redis Cache (Opsiyonel):**
```bash
# Redis kurulumu
sudo apt install redis-server

# Python Redis client
pip install redis

# Cache configuration
CACHE_SETTINGS = {
    'backend': 'redis',  # memory, redis
    'host': 'localhost',
    'port': 6379,
    'db': 0,
    'default_ttl': 300,
}
```

## 🤝 Katkıda Bulunma

### **1. Development Setup**

```bash
# Fork repository
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# Development branch
git checkout -b feature/your-feature-name

# Pre-commit hooks
pip install pre-commit
pre-commit install
```

### **2. Code Standards**

#### **Code Style:**
```bash
# Black formatting
black .

# Flake8 linting
flake8 .

# Type checking
mypy .
```

#### **Testing:**
```bash
# Test coverage
pytest --cov=. --cov-report=html

# Minimum coverage: 80%
```

### **3. Pull Request Process**

1. **Feature Branch** oluşturun
2. **Tests** yazın ve çalıştırın
3. **Documentation** güncelleyin
4. **Pull Request** oluşturun
5. **Code Review** bekleyin

### **4. Issue Reporting**

#### **Bug Report Template:**
```markdown
**Bug Description:**
Kısa açıklama

**Steps to Reproduce:**
1. Adım 1
2. Adım 2
3. Adım 3

**Expected Behavior:**
Beklenen davranış

**Actual Behavior:**
Gerçek davranış

**Environment:**
- OS: 
- Python Version:
- Bot Version:
```

## 📞 Destek

### **1. Community Support**

- 💬 **Discord:** [Trading Bot Community](https://discord.gg/tradingbot)
- 📧 **Email:** support@tradingbot.com
- 📖 **Wiki:** [GitHub Wiki](https://github.com/yourusername/trading-bot/wiki)
- 🐛 **Issues:** [GitHub Issues](https://github.com/yourusername/trading-bot/issues)

### **2. Documentation**

- 📚 **API Docs:** [API Documentation](docs/api.md)
- 🔧 **Configuration:** [Configuration Guide](docs/configuration.md)
- 🚀 **Deployment:** [Deployment Guide](docs/deployment.md)
- 🧪 **Testing:** [Testing Guide](docs/testing.md)

### **3. Professional Support**

- 💼 **Consulting:** Özel danışmanlık hizmeti
- 🛠️ **Custom Development:** Özel geliştirme
- 🏢 **Enterprise Support:** Kurumsal destek
- 📈 **Strategy Development:** Strateji geliştirme

## 📄 License

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## ⚠️ Disclaimer

**Risk Uyarısı:** Cryptocurrency trading yüksek risk içerir. Bu bot eğitim amaçlıdır ve finansal tavsiye değildir. Kendi riskinizle kullanın.

**Sorumluluk Reddi:** Geliştiriciler, bu bot'un kullanımından kaynaklanan herhangi bir mali kayıptan sorumlu değildir.

---

## 🎉 Teşekkürler

Bu projeye katkıda bulunan herkese teşekkürler:

- [@contributor1](https://github.com/contributor1) - Core development
- [@contributor2](https://github.com/contributor2) - ML models
- [@contributor3](https://github.com/contributor3) - Testing framework

---

**Made with ❤️ by the Trading Bot Team**

**⭐ Bu projeyi beğendiyseniz yıldız vermeyi unutmayın!**
