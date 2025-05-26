# 🚀 Production Deployment Checklist

**AI-Powered Cryptocurrency Trading Bot - Production Hazırlık Rehberi**

Bu rehber, trading bot'unuzu production ortamına güvenli bir şekilde deploy etmeniz için adım adım talimatlar içerir.

## 📋 Pre-Deployment Checklist

### 🔧 **1. Sistem Hazırlığı**

#### **Server Gereksinimleri:**
- [ ] **Ubuntu 20.04 LTS** veya üzeri
- [ ] **Minimum 4GB RAM** (8GB önerilen)
- [ ] **Minimum 10GB disk alanı** (SSD önerilen)
- [ ] **Stabil internet bağlantısı** (fiber önerilen)
- [ ] **Root erişimi** (sudo yetkisi)

#### **Domain ve SSL (Opsiyonel):**
- [ ] **Domain adı** satın alındı
- [ ] **DNS kayıtları** server IP'sine yönlendirildi
- [ ] **SSL sertifikası** için email adresi hazır

### 🔑 **2. API Anahtarları ve Hesaplar**

#### **Binance API:**
- [ ] **Binance hesabı** oluşturuldu
- [ ] **API anahtarları** oluşturuldu
- [ ] **IP whitelist** ayarlandı
- [ ] **Trading izinleri** verildi
- [ ] **Withdraw izni** kapatıldı (güvenlik)
- [ ] **Testnet hesabı** test için hazır

#### **Telegram Bot:**
- [ ] **@BotFather** ile bot oluşturuldu
- [ ] **Bot token** alındı
- [ ] **Chat ID** belirlendi
- [ ] **Admin ID** belirlendi
- [ ] **Bot test edildi**

### 🛡️ **3. Güvenlik Hazırlığı**

#### **API Güvenliği:**
- [ ] **API anahtarları** güvenli yerde saklandı
- [ ] **IP kısıtlaması** aktif
- [ ] **Read-only** izinler tercih edildi
- [ ] **Withdraw izinleri** kapatıldı

#### **Server Güvenliği:**
- [ ] **SSH key** authentication kuruldu
- [ ] **Password authentication** kapatıldı
- [ ] **Firewall** ayarlandı
- [ ] **Fail2ban** kuruldu (opsiyonel)

### 💰 **4. Risk Yönetimi**

#### **Trading Parametreleri:**
- [ ] **Maksimum risk** belirlendi (örn: %2 per trade)
- [ ] **Position size** stratejisi seçildi
- [ ] **Stop loss** yüzdeleri ayarlandı
- [ ] **Take profit** hedefleri belirlendi
- [ ] **Maximum drawdown** limiti ayarlandı

#### **Portföy Yönetimi:**
- [ ] **Başlangıç sermayesi** belirlendi
- [ ] **Maksimum pozisyon sayısı** ayarlandı
- [ ] **Emergency stop** koşulları belirlendi

## 🚀 Deployment Adımları

### **Adım 1: Sistem Kontrolü**

```bash
# Sistem gereksinimlerini kontrol et
sudo python3 deployment/production_deploy.py --dry-run
```

**Beklenen Çıktı:**
```
🔍 Running system requirements check...
✅ Python 3.8+: OK
✅ Git: OK
✅ Internet Connection: OK
✅ Disk Space (10GB): OK
✅ Memory (4GB): OK
✅ Root Access: OK
✅ System is ready for deployment
```

### **Adım 2: Otomatik Deployment**

#### **Basit Deployment (Local Files):**
```bash
# Mevcut dizindeki dosyaları deploy et
sudo python3 deployment/production_deploy.py
```

#### **Git Repository'den Deployment:**
```bash
# Git repository'den deploy et
sudo python3 deployment/production_deploy.py --repo-url https://github.com/yourusername/trading-bot.git
```

#### **Domain ve SSL ile Deployment:**
```bash
# Domain ve SSL ile deploy et
sudo python3 deployment/production_deploy.py \
    --repo-url https://github.com/yourusername/trading-bot.git \
    --domain yourdomain.com \
    --enable-ssl
```

### **Adım 3: Environment Konfigürasyonu**

```bash
# .env dosyasını düzenle
sudo nano /opt/trading-bot/.env
```

**Gerekli Ayarlar:**
```env
# Telegram Bot Configuration
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=123456789
TELEGRAM_ADMIN_ID=123456789

# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here
BINANCE_TESTNET=false  # Production için false

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
```

### **Adım 4: Service Başlatma**

```bash
# Service'i başlat
sudo systemctl start trading-bot

# Service durumunu kontrol et
sudo systemctl status trading-bot

# Logları kontrol et
sudo journalctl -u trading-bot -f
```

### **Adım 5: Bot Testi**

```bash
# Telegram bot'a mesaj gönder
/start
/status
/balance
/help
```

**Beklenen Yanıt:**
```
🤖 Trading Bot Started!
📊 Status: Active
💰 Balance: $1000.00
📈 Positions: 0 open
🔄 Strategies: 0 active
```

## ✅ Post-Deployment Checklist

### **1. Sistem Kontrolü**

- [ ] **Service aktif** (`systemctl status trading-bot`)
- [ ] **Loglar normal** (`journalctl -u trading-bot`)
- [ ] **Memory kullanımı** normal (`htop`)
- [ ] **Disk alanı** yeterli (`df -h`)

### **2. Bot Fonksiyonalite Testi**

- [ ] **Telegram komutları** çalışıyor
- [ ] **API bağlantısı** başarılı
- [ ] **Database** erişimi çalışıyor
- [ ] **Risk yönetimi** aktif
- [ ] **Bildirimler** geliyor

### **3. Güvenlik Kontrolü**

- [ ] **API anahtarları** şifrelenmiş
- [ ] **Log dosyaları** güvenli
- [ ] **File permissions** doğru
- [ ] **Network access** sınırlı

### **4. Monitoring Kurulumu**

- [ ] **Log rotation** aktif
- [ ] **Backup cron job** çalışıyor
- [ ] **Health checks** aktif
- [ ] **Alert system** çalışıyor

## 📊 Monitoring ve Maintenance

### **Günlük Kontroller**

```bash
# Service durumu
sudo systemctl status trading-bot

# Son loglar
sudo journalctl -u trading-bot --since "1 hour ago"

# System resources
htop
df -h
free -h

# Bot status (Telegram)
/status
/health
/performance
```

### **Haftalık Kontroller**

```bash
# Log dosyası boyutu
ls -lh /opt/trading-bot/logs/

# Database boyutu
ls -lh /opt/trading-bot/trading_bot.db

# Backup kontrolü
ls -lh /opt/trading-bot/backups/

# Performance raporu (Telegram)
/performance
/risk
```

### **Aylık Kontroller**

```bash
# System update
sudo apt update && sudo apt upgrade

# Python packages update
cd /opt/trading-bot
source venv/bin/activate
pip list --outdated

# SSL certificate renewal (otomatik)
sudo certbot renew --dry-run

# Performance optimization
python tests/debugging.py --performance-report
```

## 🔧 Troubleshooting

### **Service Başlamıyor**

```bash
# Detaylı log
sudo journalctl -u trading-bot -n 50

# Manual başlatma
cd /opt/trading-bot
sudo -u tradingbot ./venv/bin/python main.py

# Configuration test
sudo -u tradingbot ./venv/bin/python -c "from config import Config; print('Config OK')"
```

### **API Bağlantı Hatası**

```bash
# Network test
ping api.binance.com

# API key test
cd /opt/trading-bot
sudo -u tradingbot ./venv/bin/python -c "
from binance.client import Client
from config import Config
client = Client(Config.BINANCE_API_KEY, Config.BINANCE_SECRET_KEY)
print(client.get_account())
"
```

### **Telegram Bot Yanıt Vermiyor**

```bash
# Bot token test
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getMe"

# Webhook info
curl "https://api.telegram.org/bot<YOUR_TOKEN>/getWebhookInfo"

# Bot restart
sudo systemctl restart trading-bot
```

### **Yüksek Memory Kullanımı**

```bash
# Memory cleanup
cd /opt/trading-bot
sudo -u tradingbot ./venv/bin/python -c "
from utils.memory_manager import memory_manager
print(memory_manager.emergency_cleanup())
"

# Service restart
sudo systemctl restart trading-bot
```

## 🚨 Emergency Procedures

### **Acil Durdurma**

```bash
# Bot'u durdur
sudo systemctl stop trading-bot

# Telegram ile acil durdurma
/admin emergency_stop

# Tüm pozisyonları kapat (manuel)
/close_all_positions
```

### **Rollback Procedure**

```bash
# Service durdur
sudo systemctl stop trading-bot

# Backup'tan geri yükle
cd /opt/trading-bot
cp backups/trading_bot_backup_YYYYMMDD.db trading_bot.db

# Eski version'a dön
git checkout previous_version

# Service başlat
sudo systemctl start trading-bot
```

### **Data Recovery**

```bash
# Database backup
cp /opt/trading-bot/trading_bot.db /opt/trading-bot/backups/emergency_backup_$(date +%Y%m%d_%H%M%S).db

# Log backup
tar -czf /opt/trading-bot/backups/logs_backup_$(date +%Y%m%d_%H%M%S).tar.gz /opt/trading-bot/logs/

# Configuration backup
cp /opt/trading-bot/.env /opt/trading-bot/backups/env_backup_$(date +%Y%m%d_%H%M%S)
```

## 📞 Support ve Resources

### **Log Locations**
- **Application Logs:** `/opt/trading-bot/logs/trading_bot.log`
- **System Logs:** `sudo journalctl -u trading-bot`
- **Nginx Logs:** `/var/log/nginx/access.log`, `/var/log/nginx/error.log`

### **Important Files**
- **Configuration:** `/opt/trading-bot/.env`
- **Database:** `/opt/trading-bot/trading_bot.db`
- **Service File:** `/etc/systemd/system/trading-bot.service`
- **Nginx Config:** `/etc/nginx/sites-available/trading-bot`

### **Useful Commands**
```bash
# Service management
sudo systemctl {start|stop|restart|status} trading-bot

# Log monitoring
sudo journalctl -u trading-bot -f

# Performance monitoring
htop
iotop
nethogs

# Database access
sqlite3 /opt/trading-bot/trading_bot.db

# Manual bot control
cd /opt/trading-bot
sudo -u tradingbot ./venv/bin/python main.py
```

## ✅ Final Verification

### **Deployment Success Criteria**

- [ ] ✅ **Service Running:** `systemctl status trading-bot` shows "active (running)"
- [ ] ✅ **Telegram Responsive:** Bot responds to `/start` command
- [ ] ✅ **API Connected:** `/balance` command shows account balance
- [ ] ✅ **Database Working:** `/history` command shows data
- [ ] ✅ **Monitoring Active:** `/health` command shows system status
- [ ] ✅ **Logs Clean:** No ERROR messages in logs
- [ ] ✅ **Memory Normal:** Memory usage < 80%
- [ ] ✅ **Disk Space:** Available space > 5GB
- [ ] ✅ **SSL Working:** HTTPS access works (if configured)
- [ ] ✅ **Backups Scheduled:** Cron jobs are active

### **Go-Live Checklist**

- [ ] ✅ **Testnet Testing:** All features tested on testnet
- [ ] ✅ **Risk Limits:** All risk parameters configured
- [ ] ✅ **Emergency Contacts:** Admin contacts configured
- [ ] ✅ **Monitoring Setup:** Alerts and notifications active
- [ ] ✅ **Backup Strategy:** Automated backups working
- [ ] ✅ **Documentation:** All procedures documented
- [ ] ✅ **Team Training:** Team knows how to operate the bot

---

## 🎉 Congratulations!

**Your AI-Powered Cryptocurrency Trading Bot is now live in production!**

### **Next Steps:**
1. **Monitor Performance:** Keep an eye on bot performance and market conditions
2. **Optimize Strategies:** Fine-tune trading strategies based on results
3. **Scale Gradually:** Start with small amounts and scale up gradually
4. **Stay Updated:** Keep the bot and dependencies updated
5. **Learn and Improve:** Continuously improve based on experience

### **Remember:**
- 🚨 **Trading involves risk** - Never invest more than you can afford to lose
- 📊 **Monitor regularly** - Keep track of bot performance and market conditions
- 🔒 **Security first** - Regularly update and secure your system
- 📚 **Keep learning** - Stay updated with market trends and bot improvements

**Happy Trading! 🚀💰**

---

*For support and questions, refer to the README.md file or contact the development team.*
