#!/bin/bash

# Trading Bot Installation Script
# Bu script trading bot'u Ubuntu/Debian sistemlerde kurar

set -e

echo "🤖 Trading Bot Installation Starting..."

# Renk kodları
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Hata fonksiyonu
error_exit() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
    exit 1
}

# Başarı mesajı
success_msg() {
    echo -e "${GREEN}✅ $1${NC}"
}

# Uyarı mesajı
warning_msg() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

# Root kontrolü
if [[ $EUID -eq 0 ]]; then
   error_exit "Bu script root kullanıcısı ile çalıştırılmamalıdır!"
fi

# Sistem güncellemesi
echo "📦 Sistem paketleri güncelleniyor..."
sudo apt update && sudo apt upgrade -y

# Gerekli paketleri yükle
echo "📦 Gerekli paketler yükleniyor..."
sudo apt install -y python3 python3-pip python3-venv git curl wget htop

# Python versiyonu kontrolü
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
if [[ $(echo "$PYTHON_VERSION < 3.8" | bc -l) -eq 1 ]]; then
    error_exit "Python 3.8 veya üzeri gerekli. Mevcut versiyon: $PYTHON_VERSION"
fi
success_msg "Python versiyonu uygun: $PYTHON_VERSION"

# Trading bot dizini oluştur
BOT_DIR="/home/$USER/trading-bot"
if [ -d "$BOT_DIR" ]; then
    warning_msg "Trading bot dizini zaten mevcut: $BOT_DIR"
    read -p "Mevcut dizini silip yeniden kurmak istiyor musunuz? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$BOT_DIR"
        success_msg "Mevcut dizin silindi"
    else
        error_exit "Kurulum iptal edildi"
    fi
fi

# Bot dosyalarını kopyala
echo "📁 Bot dosyları kopyalanıyor..."
mkdir -p "$BOT_DIR"
cp -r . "$BOT_DIR/"
cd "$BOT_DIR"

# Virtual environment oluştur
echo "🐍 Python virtual environment oluşturuluyor..."
python3 -m venv venv
source venv/bin/activate

# Python paketlerini yükle
echo "📦 Python paketleri yükleniyor..."
pip install --upgrade pip
pip install -r requirements.txt

# Log dizini oluştur
mkdir -p logs
chmod 755 logs

# .env dosyası oluştur
if [ ! -f ".env" ]; then
    echo "⚙️ .env dosyası oluşturuluyor..."
    cp .env.example .env
    warning_msg ".env dosyasını düzenlemeyi unutmayın!"
else
    success_msg ".env dosyası zaten mevcut"
fi

# Veritabanını başlat
echo "🗄️ Veritabanı başlatılıyor..."
python -c "
from utils.database import AdvancedDatabaseManager
db = AdvancedDatabaseManager()
db.initialize()
print('✅ Veritabanı başarıyla oluşturuldu')
"

# Systemd service dosyasını kopyala
echo "🔧 Systemd service kurulumu..."
sudo cp deployment/trading-bot.service /etc/systemd/system/
sudo sed -i "s|/home/ubuntu|/home/$USER|g" /etc/systemd/system/trading-bot.service
sudo sed -i "s|User=ubuntu|User=$USER|g" /etc/systemd/system/trading-bot.service
sudo sed -i "s|Group=ubuntu|Group=$USER|g" /etc/systemd/system/trading-bot.service

# Systemd reload
sudo systemctl daemon-reload
sudo systemctl enable trading-bot.service

# Log rotation kurulumu
echo "📝 Log rotation kurulumu..."
sudo tee /etc/logrotate.d/trading-bot > /dev/null <<EOF
$BOT_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $USER $USER
    postrotate
        systemctl reload trading-bot.service > /dev/null 2>&1 || true
    endscript
}
EOF

# Scriptleri executable yap
chmod +x deployment/*.sh

# Kurulum tamamlandı
echo
echo "🎉 Trading Bot kurulumu tamamlandı!"
echo
echo "📋 Sonraki adımlar:"
echo "1. .env dosyasını düzenleyin: nano $BOT_DIR/.env"
echo "2. API anahtarlarınızı ekleyin"
echo "3. Telegram bot token'ınızı ekleyin"
echo "4. Chat ID'nizi ekleyin"
echo "5. Botu başlatın: sudo systemctl start trading-bot"
echo "6. Durumu kontrol edin: sudo systemctl status trading-bot"
echo
echo "📚 Kullanışlı komutlar:"
echo "• Bot durumu: sudo systemctl status trading-bot"
echo "• Bot başlat: sudo systemctl start trading-bot"
echo "• Bot durdur: sudo systemctl stop trading-bot"
echo "• Bot restart: sudo systemctl restart trading-bot"
echo "• Logları görüntüle: journalctl -u trading-bot -f"
echo "• Bot logları: tail -f $BOT_DIR/logs/trading_bot.log"
echo
success_msg "Kurulum başarıyla tamamlandı! 🚀"
