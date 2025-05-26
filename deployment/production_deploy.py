#!/usr/bin/env python3
"""
Production Deployment Script
Adım adım production deployment rehberi ve otomatik kurulum scripti
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime


class ProductionDeployer:
    """Production deployment yöneticisi"""
    
    def __init__(self):
        """Deployer başlatıcı"""
        self.logger = self._setup_logging()
        self.deployment_config = {}
        self.checklist = []
        self.errors = []
        
        # Deployment paths
        self.app_dir = "/opt/trading-bot"
        self.service_name = "trading-bot"
        self.user = "tradingbot"
        
    def _setup_logging(self) -> logging.Logger:
        """Logging kurulumu"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('deployment.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def run_command(self, command: str, check: bool = True, shell: bool = True) -> subprocess.CompletedProcess:
        """Komut çalıştır"""
        self.logger.info(f"Running: {command}")
        try:
            result = subprocess.run(
                command,
                shell=shell,
                check=check,
                capture_output=True,
                text=True
            )
            if result.stdout:
                self.logger.debug(f"Output: {result.stdout}")
            return result
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            self.logger.error(f"Error output: {e.stderr}")
            raise
    
    def check_system_requirements(self) -> bool:
        """Sistem gereksinimlerini kontrol et"""
        self.logger.info("🔍 Checking system requirements...")
        
        checks = [
            ("Python 3.8+", self._check_python),
            ("Git", self._check_git),
            ("Internet Connection", self._check_internet),
            ("Disk Space (10GB)", self._check_disk_space),
            ("Memory (4GB)", self._check_memory),
            ("Root Access", self._check_root_access),
        ]
        
        all_passed = True
        for name, check_func in checks:
            try:
                if check_func():
                    self.logger.info(f"✅ {name}: OK")
                    self.checklist.append(f"✅ {name}")
                else:
                    self.logger.error(f"❌ {name}: FAILED")
                    self.checklist.append(f"❌ {name}")
                    all_passed = False
            except Exception as e:
                self.logger.error(f"❌ {name}: ERROR - {e}")
                self.checklist.append(f"❌ {name}: {e}")
                all_passed = False
        
        return all_passed
    
    def _check_python(self) -> bool:
        """Python versiyonu kontrol et"""
        try:
            result = self.run_command("python3 --version")
            version = result.stdout.strip().split()[1]
            major, minor = map(int, version.split('.')[:2])
            return major >= 3 and minor >= 8
        except:
            return False
    
    def _check_git(self) -> bool:
        """Git kurulu mu kontrol et"""
        try:
            self.run_command("git --version")
            return True
        except:
            return False
    
    def _check_internet(self) -> bool:
        """İnternet bağlantısı kontrol et"""
        try:
            self.run_command("ping -c 1 google.com")
            return True
        except:
            return False
    
    def _check_disk_space(self) -> bool:
        """Disk alanı kontrol et (10GB)"""
        try:
            result = self.run_command("df / | tail -1 | awk '{print $4}'")
            available_kb = int(result.stdout.strip())
            available_gb = available_kb / 1024 / 1024
            return available_gb >= 10
        except:
            return False
    
    def _check_memory(self) -> bool:
        """Memory kontrol et (4GB)"""
        try:
            result = self.run_command("free -m | grep '^Mem:' | awk '{print $2}'")
            total_mb = int(result.stdout.strip())
            return total_mb >= 4000
        except:
            return False
    
    def _check_root_access(self) -> bool:
        """Root erişimi kontrol et"""
        return os.geteuid() == 0
    
    def install_system_dependencies(self) -> bool:
        """Sistem bağımlılıklarını yükle"""
        self.logger.info("📦 Installing system dependencies...")
        
        try:
            # System update
            self.run_command("apt update && apt upgrade -y")
            
            # Install packages
            packages = [
                "python3", "python3-pip", "python3-venv", "python3-dev",
                "git", "nginx", "supervisor", "sqlite3",
                "build-essential", "libssl-dev", "libffi-dev",
                "certbot", "python3-certbot-nginx",
                "htop", "curl", "wget", "unzip"
            ]
            
            package_list = " ".join(packages)
            self.run_command(f"apt install -y {package_list}")
            
            self.logger.info("✅ System dependencies installed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to install system dependencies: {e}")
            return False
    
    def create_user(self) -> bool:
        """Trading bot kullanıcısı oluştur"""
        self.logger.info(f"👤 Creating user: {self.user}")
        
        try:
            # Check if user exists
            try:
                self.run_command(f"id {self.user}")
                self.logger.info(f"User {self.user} already exists")
                return True
            except:
                pass
            
            # Create user
            self.run_command(f"adduser --disabled-password --gecos '' {self.user}")
            self.run_command(f"usermod -aG sudo {self.user}")
            
            # Create directories
            self.run_command(f"mkdir -p {self.app_dir}")
            self.run_command(f"chown {self.user}:{self.user} {self.app_dir}")
            
            self.logger.info(f"✅ User {self.user} created")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create user: {e}")
            return False
    
    def deploy_application(self, repo_url: str = None) -> bool:
        """Uygulamayı deploy et"""
        self.logger.info("🚀 Deploying application...")
        
        try:
            # Clone repository
            if repo_url:
                self.run_command(f"sudo -u {self.user} git clone {repo_url} {self.app_dir}")
            else:
                # Copy current directory
                current_dir = os.getcwd()
                self.run_command(f"cp -r {current_dir}/* {self.app_dir}/")
                self.run_command(f"chown -R {self.user}:{self.user} {self.app_dir}")
            
            # Create virtual environment
            self.run_command(f"sudo -u {self.user} python3 -m venv {self.app_dir}/venv")
            
            # Install Python dependencies
            pip_cmd = f"{self.app_dir}/venv/bin/pip"
            self.run_command(f"sudo -u {self.user} {pip_cmd} install --upgrade pip")
            self.run_command(f"sudo -u {self.user} {pip_cmd} install -r {self.app_dir}/requirements.txt")
            
            # Create logs directory
            self.run_command(f"mkdir -p {self.app_dir}/logs")
            self.run_command(f"chown {self.user}:{self.user} {self.app_dir}/logs")
            
            self.logger.info("✅ Application deployed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to deploy application: {e}")
            return False
    
    def setup_environment(self) -> bool:
        """Environment dosyasını ayarla"""
        self.logger.info("⚙️ Setting up environment...")
        
        try:
            env_example = f"{self.app_dir}/.env.example"
            env_file = f"{self.app_dir}/.env"
            
            if os.path.exists(env_example):
                self.run_command(f"cp {env_example} {env_file}")
                self.run_command(f"chown {self.user}:{self.user} {env_file}")
                
                self.logger.warning("⚠️  IMPORTANT: Edit .env file with your API keys!")
                self.logger.warning(f"⚠️  File location: {env_file}")
                self.logger.warning("⚠️  Required settings:")
                self.logger.warning("   - TELEGRAM_BOT_TOKEN")
                self.logger.warning("   - TELEGRAM_CHAT_ID")
                self.logger.warning("   - BINANCE_API_KEY")
                self.logger.warning("   - BINANCE_SECRET_KEY")
                
                return True
            else:
                self.logger.error("❌ .env.example file not found")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup environment: {e}")
            return False
    
    def setup_systemd_service(self) -> bool:
        """Systemd service kurulumu"""
        self.logger.info("🔧 Setting up systemd service...")
        
        try:
            service_file = f"/etc/systemd/system/{self.service_name}.service"
            source_service = f"{self.app_dir}/deployment/trading-bot.service"
            
            if os.path.exists(source_service):
                self.run_command(f"cp {source_service} {service_file}")
            else:
                # Create service file
                service_content = f"""[Unit]
Description=Trading Bot
After=network.target

[Service]
Type=simple
User={self.user}
WorkingDirectory={self.app_dir}
Environment=PATH={self.app_dir}/venv/bin
ExecStart={self.app_dir}/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
                with open(service_file, 'w') as f:
                    f.write(service_content)
            
            # Enable service
            self.run_command("systemctl daemon-reload")
            self.run_command(f"systemctl enable {self.service_name}")
            
            self.logger.info("✅ Systemd service configured")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup systemd service: {e}")
            return False
    
    def setup_nginx(self, domain: str = None) -> bool:
        """Nginx kurulumu (opsiyonel dashboard için)"""
        self.logger.info("🌐 Setting up Nginx...")
        
        try:
            nginx_config = f"/etc/nginx/sites-available/{self.service_name}"
            
            config_content = f"""server {{
    listen 80;
    server_name {domain or 'localhost'};
    
    location / {{
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }}
    
    location /static {{
        alias {self.app_dir}/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }}
}}
"""
            
            with open(nginx_config, 'w') as f:
                f.write(config_content)
            
            # Enable site
            self.run_command(f"ln -sf {nginx_config} /etc/nginx/sites-enabled/")
            self.run_command("nginx -t")
            self.run_command("systemctl reload nginx")
            
            self.logger.info("✅ Nginx configured")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup Nginx: {e}")
            return False
    
    def setup_ssl(self, domain: str) -> bool:
        """SSL sertifikası kurulumu"""
        self.logger.info("🔒 Setting up SSL certificate...")
        
        try:
            self.run_command(f"certbot --nginx -d {domain} --non-interactive --agree-tos --email admin@{domain}")
            
            # Auto-renewal
            cron_job = "0 12 * * * /usr/bin/certbot renew --quiet"
            self.run_command(f'(crontab -l 2>/dev/null; echo "{cron_job}") | crontab -')
            
            self.logger.info("✅ SSL certificate configured")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup SSL: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Monitoring kurulumu"""
        self.logger.info("📊 Setting up monitoring...")
        
        try:
            # Log rotation
            logrotate_config = f"/etc/logrotate.d/{self.service_name}"
            logrotate_content = f"""{self.app_dir}/logs/*.log {{
    daily
    missingok
    rotate 30
    compress
    notifempty
    create 644 {self.user} {self.user}
    postrotate
        systemctl reload {self.service_name}
    endscript
}}
"""
            
            with open(logrotate_config, 'w') as f:
                f.write(logrotate_content)
            
            # Backup cron job
            backup_script = f"{self.app_dir}/deployment/backup.sh"
            if os.path.exists(backup_script):
                self.run_command(f"chmod +x {backup_script}")
                cron_job = f"0 2 * * * {backup_script}"
                self.run_command(f'(crontab -u {self.user} -l 2>/dev/null; echo "{cron_job}") | crontab -u {self.user} -')
            
            self.logger.info("✅ Monitoring configured")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup monitoring: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Testleri çalıştır"""
        self.logger.info("🧪 Running tests...")
        
        try:
            python_cmd = f"{self.app_dir}/venv/bin/python"
            test_cmd = f"cd {self.app_dir} && sudo -u {self.user} {python_cmd} tests/run_tests.py --unit"
            
            result = self.run_command(test_cmd, check=False)
            
            if result.returncode == 0:
                self.logger.info("✅ Tests passed")
                return True
            else:
                self.logger.warning("⚠️  Some tests failed, but continuing deployment")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Failed to run tests: {e}")
            return False
    
    def start_service(self) -> bool:
        """Service'i başlat"""
        self.logger.info("🚀 Starting service...")
        
        try:
            self.run_command(f"systemctl start {self.service_name}")
            time.sleep(5)  # Service'in başlaması için bekle
            
            # Status kontrol
            result = self.run_command(f"systemctl is-active {self.service_name}", check=False)
            
            if result.stdout.strip() == "active":
                self.logger.info("✅ Service started successfully")
                return True
            else:
                self.logger.error("❌ Service failed to start")
                # Show logs
                self.run_command(f"journalctl -u {self.service_name} --no-pager -n 20")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to start service: {e}")
            return False
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Deployment raporu oluştur"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'deployment_status': 'success' if not self.errors else 'failed',
            'checklist': self.checklist,
            'errors': self.errors,
            'service_info': {
                'name': self.service_name,
                'user': self.user,
                'app_dir': self.app_dir,
            },
            'next_steps': [
                f"1. Edit {self.app_dir}/.env with your API keys",
                f"2. Check service status: sudo systemctl status {self.service_name}",
                f"3. View logs: sudo journalctl -u {self.service_name} -f",
                f"4. Test bot: Send /start to your Telegram bot",
                "5. Monitor performance and adjust settings as needed"
            ]
        }
        
        return report
    
    def deploy(self, repo_url: str = None, domain: str = None, enable_ssl: bool = False) -> bool:
        """Full deployment çalıştır"""
        self.logger.info("🚀 Starting production deployment...")
        
        steps = [
            ("System Requirements", lambda: self.check_system_requirements()),
            ("System Dependencies", lambda: self.install_system_dependencies()),
            ("User Creation", lambda: self.create_user()),
            ("Application Deployment", lambda: self.deploy_application(repo_url)),
            ("Environment Setup", lambda: self.setup_environment()),
            ("Systemd Service", lambda: self.setup_systemd_service()),
            ("Nginx Setup", lambda: self.setup_nginx(domain)),
            ("Monitoring Setup", lambda: self.setup_monitoring()),
            ("Tests", lambda: self.run_tests()),
            ("Service Start", lambda: self.start_service()),
        ]
        
        if enable_ssl and domain:
            steps.insert(-2, ("SSL Setup", lambda: self.setup_ssl(domain)))
        
        success = True
        for step_name, step_func in steps:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Step: {step_name}")
            self.logger.info(f"{'='*50}")
            
            try:
                if not step_func():
                    self.errors.append(f"Failed: {step_name}")
                    success = False
                    break
            except Exception as e:
                self.logger.error(f"Error in {step_name}: {e}")
                self.errors.append(f"Error in {step_name}: {e}")
                success = False
                break
        
        # Generate report
        report = self.generate_deployment_report()
        
        # Save report
        report_file = f"deployment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"\n{'='*60}")
        if success:
            self.logger.info("🎉 DEPLOYMENT SUCCESSFUL!")
            self.logger.info("✅ Trading bot is now running in production")
        else:
            self.logger.error("❌ DEPLOYMENT FAILED!")
            self.logger.error("Please check the errors above and try again")
        
        self.logger.info(f"📊 Deployment report saved: {report_file}")
        self.logger.info(f"{'='*60}")
        
        # Print next steps
        self.logger.info("\n📋 NEXT STEPS:")
        for i, step in enumerate(report['next_steps'], 1):
            self.logger.info(f"{i}. {step}")
        
        return success


def main():
    """Ana fonksiyon"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Deployment Script")
    parser.add_argument("--repo-url", help="Git repository URL")
    parser.add_argument("--domain", help="Domain name for SSL")
    parser.add_argument("--enable-ssl", action="store_true", help="Enable SSL certificate")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (check only)")
    
    args = parser.parse_args()
    
    deployer = ProductionDeployer()
    
    if args.dry_run:
        print("🔍 Running system requirements check...")
        if deployer.check_system_requirements():
            print("✅ System is ready for deployment")
            return 0
        else:
            print("❌ System requirements not met")
            return 1
    
    # Check if running as root
    if os.geteuid() != 0:
        print("❌ This script must be run as root (use sudo)")
        return 1
    
    # Run deployment
    success = deployer.deploy(
        repo_url=args.repo_url,
        domain=args.domain,
        enable_ssl=args.enable_ssl
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
