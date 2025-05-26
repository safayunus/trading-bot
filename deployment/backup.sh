#!/bin/bash

# Trading Bot Backup Script
# Veritabanı ve konfigürasyon dosyalarını yedekler

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

# Konfigürasyon
BOT_DIR="/home/$USER/trading-bot"
BACKUP_DIR="/home/$USER/trading-bot-backups"
DATE=$(date '+%Y%m%d_%H%M%S')
BACKUP_NAME="trading_bot_backup_$DATE"
BACKUP_PATH="$BACKUP_DIR/$BACKUP_NAME"

echo "💾 Trading Bot Backup Script"
echo "============================"
echo "Tarih: $(date '+%Y-%m-%d %H:%M:%S')"
echo

# Backup dizini oluştur
info_msg "Backup dizini hazırlanıyor..."
mkdir -p "$BACKUP_DIR"
mkdir -p "$BACKUP_PATH"

# Bot dizini kontrolü
if [ ! -d "$BOT_DIR" ]; then
    error_msg "Bot dizini bulunamadı: $BOT_DIR"
    exit 1
fi

cd "$BOT_DIR"

# Veritabanı yedekleme
info_msg "Veritabanı yedekleniyor..."
if [ -f "trading_bot.db" ]; then
    cp "trading_bot.db" "$BACKUP_PATH/trading_bot.db"
    success_msg "Veritabanı yedeklendi"
else
    warning_msg "Veritabanı dosyası bulunamadı"
fi

# Konfigürasyon dosyaları
info_msg "Konfigürasyon dosyaları yedekleniyor..."
if [ -f ".env" ]; then
    cp ".env" "$BACKUP_PATH/.env"
    success_msg ".env dosyası yedeklendi"
else
    warning_msg ".env dosyası bulunamadı"
fi

# Log dosyaları (son 7 günlük)
info_msg "Log dosyaları yedekleniyor..."
if [ -d "logs" ]; then
    mkdir -p "$BACKUP_PATH/logs"
    
    # Son 7 günün loglarını al
    find logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_PATH/logs/" \; 2>/dev/null || true
    
    # Log sayısını kontrol et
    LOG_COUNT=$(find "$BACKUP_PATH/logs" -name "*.log" | wc -l)
    if [ "$LOG_COUNT" -gt 0 ]; then
        success_msg "$LOG_COUNT log dosyası yedeklendi"
    else
        warning_msg "Yedeklenecek log dosyası bulunamadı"
    fi
else
    warning_msg "Log dizini bulunamadı"
fi

# Deployment scriptleri
info_msg "Deployment scriptleri yedekleniyor..."
if [ -d "deployment" ]; then
    cp -r "deployment" "$BACKUP_PATH/"
    success_msg "Deployment scriptleri yedeklendi"
fi

# Backup bilgi dosyası oluştur
info_msg "Backup bilgi dosyası oluşturuluyor..."
cat > "$BACKUP_PATH/backup_info.txt" << EOF
Trading Bot Backup Information
==============================
Backup Date: $(date '+%Y-%m-%d %H:%M:%S')
Bot Directory: $BOT_DIR
Backup Directory: $BACKUP_PATH
System: $(uname -a)
Python Version: $(python3 --version 2>/dev/null || echo "Not found")

Files Included:
- trading_bot.db (database)
- .env (configuration)
- logs/ (recent log files)
- deployment/ (scripts)

Bot Status at Backup Time:
$(systemctl is-active trading-bot 2>/dev/null && echo "Running" || echo "Stopped")

Disk Usage:
$(du -sh "$BOT_DIR" 2>/dev/null || echo "Unknown")
EOF

# Backup'ı sıkıştır
info_msg "Backup sıkıştırılıyor..."
cd "$BACKUP_DIR"
tar -czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"

if [ $? -eq 0 ]; then
    # Sıkıştırılmış dosya boyutu
    BACKUP_SIZE=$(du -sh "${BACKUP_NAME}.tar.gz" | cut -f1)
    success_msg "Backup sıkıştırıldı: ${BACKUP_NAME}.tar.gz ($BACKUP_SIZE)"
    
    # Geçici dizini sil
    rm -rf "$BACKUP_NAME"
    
    # Backup dosya yolu
    FINAL_BACKUP="$BACKUP_DIR/${BACKUP_NAME}.tar.gz"
    
else
    error_msg "Backup sıkıştırma hatası!"
    exit 1
fi

# Eski backup'ları temizle (30 günden eski)
info_msg "Eski backup'lar temizleniyor..."
find "$BACKUP_DIR" -name "trading_bot_backup_*.tar.gz" -mtime +30 -delete 2>/dev/null || true

# Kalan backup sayısı
BACKUP_COUNT=$(find "$BACKUP_DIR" -name "trading_bot_backup_*.tar.gz" | wc -l)
info_msg "Toplam backup sayısı: $BACKUP_COUNT"

echo
success_msg "Backup işlemi tamamlandı! 🎉"
echo
info_msg "Backup Detayları:"
echo "   Dosya: $FINAL_BACKUP"
echo "   Boyut: $BACKUP_SIZE"
echo "   Tarih: $(date '+%Y-%m-%d %H:%M:%S')"
echo
info_msg "Backup'ı geri yüklemek için:"
echo "   1. Bot'u durdur: sudo systemctl stop trading-bot"
echo "   2. Backup'ı çıkart: tar -xzf $FINAL_BACKUP -C /tmp/"
echo "   3. Dosyaları kopyala: cp -r /tmp/$BACKUP_NAME/* $BOT_DIR/"
echo "   4. Bot'u başlat: sudo systemctl start trading-bot"
