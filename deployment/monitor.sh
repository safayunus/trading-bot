#!/bin/bash

# Trading Bot Monitoring Script
# Bot'un durumunu izler ve raporlar

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Mesaj fonksiyonları
info_msg() {
    echo -e "${BLUE}ℹ️ $1${NC}"
}

success_msg() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning_msg() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

error_msg() {
    echo -e "${RED}❌ $1${NC}"
}

# Header
echo "📊 Trading Bot Monitor"
echo "====================="
echo "Tarih: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# Bot durumu
info_msg "Bot Service Durumu:"
if systemctl is-active --quiet trading-bot; then
    success_msg "Bot çalışıyor ✓"
    
    # Uptime bilgisi
    UPTIME=$(systemctl show trading-bot --property=ActiveEnterTimestamp --value)
    if [ ! -z "$UPTIME" ]; then
        echo "   Başlama zamanı: $UPTIME"
    fi
    
    # Memory kullanımı
    PID=$(systemctl show trading-bot --property=MainPID --value)
    if [ "$PID" != "0" ] && [ ! -z "$PID" ]; then
        MEMORY=$(ps -p $PID -o rss= 2>/dev/null | awk '{print $1/1024 " MB"}')
        CPU=$(ps -p $PID -o %cpu= 2>/dev/null | awk '{print $1"%"}')
        echo "   Memory: $MEMORY"
        echo "   CPU: $CPU"
    fi
else
    error_msg "Bot çalışmıyor ✗"
    
    # Son hata logları
    echo "   Son hata logları:"
    journalctl -u trading-bot --no-pager -n 5 --since "1 hour ago" | grep -i error || echo "   Hata bulunamadı"
fi

echo

# Disk kullanımı
info_msg "Disk Kullanımı:"
BOT_DIR="/home/$USER/trading-bot"
if [ -d "$BOT_DIR" ]; then
    DISK_USAGE=$(du -sh "$BOT_DIR" 2>/dev/null | cut -f1)
    echo "   Bot dizini: $DISK_USAGE"
    
    # Log dosyası boyutu
    if [ -d "$BOT_DIR/logs" ]; then
        LOG_SIZE=$(du -sh "$BOT_DIR/logs" 2>/dev/null | cut -f1)
        echo "   Log dosyaları: $LOG_SIZE"
    fi
    
    # Database boyutu
    if [ -f "$BOT_DIR/trading_bot.db" ]; then
        DB_SIZE=$(du -sh "$BOT_DIR/trading_bot.db" 2>/dev/null | cut -f1)
        echo "   Veritabanı: $DB_SIZE"
    fi
fi

echo

# Network bağlantısı
info_msg "Network Bağlantısı:"
if ping -c 1 google.com &> /dev/null; then
    success_msg "Internet bağlantısı aktif ✓"
else
    error_msg "Internet bağlantısı yok ✗"
fi

# Binance API test
if curl -s "https://api.binance.com/api/v3/ping" &> /dev/null; then
    success_msg "Binance API erişilebilir ✓"
else
    warning_msg "Binance API erişilemiyor ⚠️"
fi

echo

# Son loglar
info_msg "Son Sistem Logları (Son 5 kayıt):"
echo "----------------------------------------"
journalctl -u trading-bot --no-pager -n 5 --output=short
echo "----------------------------------------"

echo

# Bot logları (eğer varsa)
if [ -f "$BOT_DIR/logs/trading_bot.log" ]; then
    info_msg "Son Bot Logları (Son 5 satır):"
    echo "----------------------------------------"
    tail -n 5 "$BOT_DIR/logs/trading_bot.log" 2>/dev/null || echo "Log dosyası okunamadı"
    echo "----------------------------------------"
fi

echo

# Sistem kaynakları
info_msg "Sistem Kaynakları:"
echo "   CPU Kullanımı: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "   Memory Kullanımı: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "   Disk Kullanımı: $(df -h / | awk 'NR==2{printf "%s", $5}')"
echo "   Load Average: $(uptime | awk -F'load average:' '{print $2}')"

echo

# Öneriler
info_msg "Hızlı Komutlar:"
echo "   Bot durumu: sudo systemctl status trading-bot"
echo "   Bot restart: sudo systemctl restart trading-bot"
echo "   Canlı loglar: journalctl -u trading-bot -f"
echo "   Bot logları: tail -f $BOT_DIR/logs/trading_bot.log"

echo
success_msg "Monitoring tamamlandı!"
