# Trading Bot Deployment Guide

Bu rehber, cryptocurrency trading bot'unun Ubuntu/Debian sistemlerde nasıl kurulacağını ve yönetileceğini açıklar.

## 📋 Gereksinimler

### Sistem Gereksinimleri
- **İşletim Sistemi:** Ubuntu 20.04+ veya Debian 11+
- **Python:** 3.8 veya üzeri
- **RAM:** Minimum 1GB (2GB önerilen)
- **Disk:** Minimum 2GB boş alan
- **Network:** Stabil internet bağlantısı

### API Gereksinimleri
- **Telegram Bot Token:** [@BotFather](https://t.me/botfather)'dan alınacak
- **Telegram Chat ID:** Bot ile konuşacağınız chat ID
- **Binance API Keys:** [Binance](https://www.binance.com/en/my/settings/api-management) hesabından

## 🚀 Hızlı Kurulum

### 1. Otomatik Kurulum
```bash
# Repository'yi klonlayın
git clone <repository-url> trading-bot
cd trading-bot

# Kurulum scriptini çalıştırın
chmod +x deployment/install.sh
./deployment/install.sh
```

### 2. Konfigürasyon
```bash
# .env dosyasını düzenleyin
nano /home/$USER/trading-bot/.env

# Gerekli değerleri girin:
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
TELEGRAM_ADMIN_IDS=your_admin_id_here
BINANCE_API_KEY=your_api_key_here
BINANCE_SECRET_KEY=your_secret_key_here
```

### 3. Bot'u Başlatın
```bash
# Bot'u başlat
sudo systemctl start trading-bot

# Durumu kontrol et
sudo systemctl status trading-bot

# Otomatik başlatmayı etkinleştir
sudo systemctl enable trading-bot
```

## 🔧 Manuel Kurulum

### 1. Sistem Hazırlığı
```bash
# Sistem güncellemesi
sudo apt update && sudo apt upgrade -y

# Gerekli paketler
sudo apt install -y python3 python3-pip python3-venv git curl wget htop
```

### 2. Bot Kurulumu
```bash
# Bot dizini oluştur
mkdir -p /home/$USER/trading-bot
cd /home/$USER/trading-bot

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Python paketleri
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Veritabanı Kurulumu
```bash
# Veritabanını başlat
python -c "
from utils.database import AdvancedDatabaseManager
db = AdvancedDatabaseManager()
db.initialize()
print('Database initialized successfully')
"
```

### 4. Systemd Service
```bash
# Service dosyasını kopyala
sudo cp deployment/trading-bot.service /etc/systemd/system/

# Kullanıcı adını güncelle
sudo sed -i "s|/home/ubuntu|/home/$USER|g" /etc/systemd/system/trading-bot.service
sudo sed -i "s|User=ubuntu|User=$USER|g" /etc/systemd/system/trading-bot.service
sudo sed -i "s|Group=ubuntu|Group=$USER|g" /etc/systemd/system/trading-bot.service

# Systemd reload
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
```

## ⚙️ Konfigürasyon

### Environment Variables (.env)

#### Zorunlu Ayarlar
```bash
# Telegram Bot
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
TELEGRAM_ADMIN_IDS=123456789,987654321

# Binance API
BINANCE_API_KEY=your_64_character_api_key
BINANCE_SECRET_KEY=your_64_character_secret_key
BINANCE_TESTNET=true  # false for live trading
```

#### İsteğe Bağlı Ayarlar
```bash
# Trading
INITIAL_CAPITAL=10000.0
MAX_RISK_PER_TRADE=0.02
MAX_DAILY_LOSS=0.05
TRADING_ENABLED=true

# Logging
LOG_LEVEL=INFO
LOG_FILE_PATH=logs/trading_bot.log

# Reports
DAILY_REPORTS_ENABLED=true
DAILY_REPORT_TIME=09:00
```

### Telegram Bot Token Alma

1. [@BotFather](https://t.me/botfather) ile konuşun
2. `/newbot` komutunu gönderin
3. Bot adını ve kullanıcı adını belirleyin
4. Verilen token'ı `.env` dosyasına ekleyin

### Chat ID Bulma

1. Bot'unuza bir mesaj gönderin
2. Bu URL'yi ziyaret edin: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
3. `chat.id` değerini bulun ve `.env` dosyasına ekleyin

## 🎮 Bot Yönetimi

### Temel Komutlar
```bash
# Bot durumu
sudo systemctl status trading-bot

# Bot başlat
sudo systemctl start trading-bot

# Bot durdur
sudo systemctl stop trading-bot

# Bot restart
sudo systemctl restart trading-bot

# Logları görüntüle
journalctl -u trading-bot -f

# Bot logları
tail -f /home/$USER/trading-bot/logs/trading_bot.log
```

### Yardımcı Scriptler
```bash
# Bot restart (güvenli)
./deployment/restart.sh

# Bot monitoring
./deployment/monitor.sh

# Backup oluştur
./deployment/backup.sh
```

## 📊 Monitoring

### Sistem Durumu
```bash
# Bot durumu kontrol
./deployment/monitor.sh

# Sistem kaynakları
htop

# Disk kullanımı
df -h

# Log boyutları
du -sh /home/$USER/trading-bot/logs/
```

### Telegram Komutları
```
/status - Bot durumu
/balance - Portföy bakiyesi
/positions - Açık pozisyonlar
/pnl - Kar/zarar raporu
/report - Günlük rapor
/performance - Performans özeti
/realtime - Anlık durum
/help - Komut listesi
```

## 🔒 Güvenlik

### Temel Güvenlik Önlemleri

1. **API Keys Güvenliği**
   ```bash
   # .env dosyası izinleri
   chmod 600 /home/$USER/trading-bot/.env
   
   # Sadece okuma izni
   chown $USER:$USER /home/$USER/trading-bot/.env
   ```

2. **Firewall Ayarları**
   ```bash
   # UFW etkinleştir
   sudo ufw enable
   
   # SSH izni
   sudo ufw allow ssh
   
   # Gereksiz portları kapat
   sudo ufw default deny incoming
   sudo ufw default allow outgoing
   ```

3. **User Authentication**
   - Sadece belirtilen Telegram kullanıcıları bot'u kullanabilir
   - Chat ID kontrolü ile erişim sınırlandırması
   - Rate limiting ile spam koruması

### Güvenlik Kontrolleri
```bash
# Environment variables kontrolü
python3 -c "
from deployment.security import check_security_setup
if check_security_setup():
    print('✅ Security setup OK')
else:
    print('❌ Security issues found')
"
```

## 💾 Backup & Recovery

### Otomatik Backup
```bash
# Backup oluştur
./deployment/backup.sh

# Backup'lar
ls -la /home/$USER/trading-bot-backups/
```

### Manuel Backup
```bash
# Veritabanı backup
cp /home/$USER/trading-bot/trading_bot.db /backup/location/

# Konfigürasyon backup
cp /home/$USER/trading-bot/.env /backup/location/

# Log backup
tar -czf logs_backup.tar.gz /home/$USER/trading-bot/logs/
```

### Recovery
```bash
# Bot'u durdur
sudo systemctl stop trading-bot

# Backup'ı geri yükle
tar -xzf trading_bot_backup_YYYYMMDD_HHMMSS.tar.gz -C /tmp/
cp -r /tmp/trading_bot_backup_YYYYMMDD_HHMMSS/* /home/$USER/trading-bot/

# Bot'u başlat
sudo systemctl start trading-bot
```

## 📝 Log Yönetimi

### Log Rotation
Log rotation otomatik olarak kurulur:
- **Günlük rotation:** Her gün yeni log dosyası
- **30 gün saklama:** Eski loglar otomatik silinir
- **Sıkıştırma:** Eski loglar gzip ile sıkıştırılır

### Log Dosyaları
```bash
# Bot logları
/home/$USER/trading-bot/logs/trading_bot.log

# System logları
journalctl -u trading-bot

# Error logları
journalctl -u trading-bot -p err
```

## 🚨 Troubleshooting

### Yaygın Sorunlar

#### 1. Bot Başlamıyor
```bash
# Hata loglarını kontrol et
journalctl -u trading-bot -n 50

# Konfigürasyon kontrol
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('TELEGRAM_BOT_TOKEN:', 'OK' if os.getenv('TELEGRAM_BOT_TOKEN') else 'MISSING')
print('BINANCE_API_KEY:', 'OK' if os.getenv('BINANCE_API_KEY') else 'MISSING')
"
```

#### 2. Telegram Bağlantı Sorunu
```bash
# Network testi
curl -s "https://api.telegram.org/bot<YOUR_TOKEN>/getMe"

# Bot token testi
python3 -c "
import requests
token = 'YOUR_BOT_TOKEN'
response = requests.get(f'https://api.telegram.org/bot{token}/getMe')
print(response.json())
"
```

#### 3. Binance API Sorunu
```bash
# API testi
curl -s "https://api.binance.com/api/v3/ping"

# API key testi
python3 -c "
from binance.client import Client
client = Client('API_KEY', 'SECRET_KEY', testnet=True)
print(client.ping())
"
```

#### 4. Veritabanı Sorunu
```bash
# Veritabanı kontrolü
python3 -c "
from utils.database import AdvancedDatabaseManager
db = AdvancedDatabaseManager()
try:
    db.initialize()
    print('✅ Database OK')
except Exception as e:
    print(f'❌ Database Error: {e}')
"
```

### Performance Sorunları

#### Yüksek Memory Kullanımı
```bash
# Memory kullanımı kontrol
ps aux | grep python

# Bot restart
./deployment/restart.sh
```

#### Yüksek CPU Kullanımı
```bash
# CPU kullanımı kontrol
top -p $(pgrep -f "python.*main.py")

# Log seviyesini düşür
# .env dosyasında: LOG_LEVEL=WARNING
```

## 📞 Destek

### Log Toplama
Sorun yaşadığınızda şu bilgileri toplayın:

```bash
# Sistem bilgisi
uname -a
python3 --version
pip list | grep -E "(telegram|binance|pandas|numpy)"

# Bot durumu
sudo systemctl status trading-bot

# Son loglar
journalctl -u trading-bot -n 100 --no-pager

# Bot logları
tail -n 100 /home/$USER/trading-bot/logs/trading_bot.log

# Disk kullanımı
df -h
du -sh /home/$USER/trading-bot/
```

### Güncelleme
```bash
# Kodu güncelle
cd /home/$USER/trading-bot
git pull origin main

# Paketleri güncelle
source venv/bin/activate
pip install -r requirements.txt --upgrade

# Bot restart
./deployment/restart.sh
```

## 📚 Ek Kaynaklar

- [Telegram Bot API](https://core.telegram.org/bots/api)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Python-Telegram-Bot](https://python-telegram-bot.readthedocs.io/)
- [Systemd Service Management](https://www.freedesktop.org/software/systemd/man/systemd.service.html)

---

**⚠️ Önemli Uyarılar:**
- Bu bot gerçek para ile trading yapar
- Test ortamında deneyip sonra canlıya geçin
- Risk yönetimi ayarlarını dikkatlice yapın
- API anahtarlarınızı kimseyle paylaşmayın
- Düzenli backup almayı unutmayın

**🚀 İyi Trading'ler!**
