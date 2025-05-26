#!/bin/bash

# Trading Bot Restart Script
# Bot'u güvenli bir şekilde yeniden başlatır

set -e

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

echo "🔄 Trading Bot Restart Script"
echo "=============================="

# Bot durumunu kontrol et
info_msg "Bot durumu kontrol ediliyor..."
if systemctl is-active --quiet trading-bot; then
    info_msg "Bot şu anda çalışıyor"
    
    # Graceful shutdown dene
    info_msg "Bot durduruluyor..."
    sudo systemctl stop trading-bot
    
    # Durmasını bekle
    sleep 3
    
    # Durduğunu kontrol et
    if systemctl is-active --quiet trading-bot; then
        warning_msg "Bot hala çalışıyor, zorla durduruluyor..."
        sudo systemctl kill trading-bot
        sleep 2
    fi
    
    success_msg "Bot başarıyla durduruldu"
else
    info_msg "Bot zaten durmuş durumda"
fi

# Systemd konfigürasyonunu yeniden yükle
info_msg "Systemd konfigürasyonu yeniden yükleniyor..."
sudo systemctl daemon-reload

# Bot'u başlat
info_msg "Bot başlatılıyor..."
sudo systemctl start trading-bot

# Başlamasını bekle
sleep 5

# Durumu kontrol et
if systemctl is-active --quiet trading-bot; then
    success_msg "Bot başarıyla başlatıldı! 🚀"
    
    # Son logları göster
    info_msg "Son loglar:"
    echo "----------------------------------------"
    journalctl -u trading-bot --no-pager -n 10
    echo "----------------------------------------"
    
    # Durum bilgisi
    echo
    info_msg "Bot durumu:"
    sudo systemctl status trading-bot --no-pager -l
    
else
    error_msg "Bot başlatılamadı!"
    echo
    error_msg "Hata logları:"
    journalctl -u trading-bot --no-pager -n 20
    exit 1
fi

echo
success_msg "Restart işlemi tamamlandı!"
info_msg "Logları takip etmek için: journalctl -u trading-bot -f"
