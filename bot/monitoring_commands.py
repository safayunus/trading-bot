"""
Monitoring and Reporting Commands
Monitoring ve raporlama komutları
"""

import logging
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Any, List
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ContextTypes

if TYPE_CHECKING:
    from .telegram_handler import TelegramBot
    from ..utils.monitoring import MonitoringSystem
    from .daily_reporter import DailyReporter

class MonitoringCommandHandler:
    """Monitoring ve rapor komutları"""
    
    def __init__(self, telegram_bot: 'TelegramBot', monitoring_system: 'MonitoringSystem', 
                 daily_reporter: 'DailyReporter'):
        """
        Monitoring komut işleyici başlatıcı
        
        Args:
            telegram_bot: TelegramBot instance
            monitoring_system: MonitoringSystem instance
            daily_reporter: DailyReporter instance
        """
        self.telegram_bot = telegram_bot
        self.monitoring = monitoring_system
        self.daily_reporter = daily_reporter
        self.logger = logging.getLogger(__name__)
    
    async def _check_authorization(self, update: Update) -> bool:
        """Kullanıcı yetkilendirme kontrolü"""
        user_id = update.effective_user.id
        
        if not self.telegram_bot.is_authorized_user(user_id):
            await update.message.reply_text(
                "❌ <b>Unauthorized Access</b>\n\n"
                "You are not authorized to use this bot.\n"
                "Please contact the administrator.",
                parse_mode='HTML'
            )
            return False
        
        return True
    
    async def report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Report komutu - Günlük rapor gönder"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Günlük rapor oluştur
            report = self.monitoring.generate_daily_report()
            
            # Telegram formatına çevir
            telegram_message = self.monitoring.format_report_for_telegram(report)
            
            await update.message.reply_text(
                telegram_message,
                parse_mode='HTML',
                reply_markup=self._get_report_keyboard()
            )
            
        except Exception as e:
            self.logger.error(f"Report komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Rapor oluşturulamadı. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    async def performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Performance komutu - Performans özeti"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Argüman kontrolü (kaç günlük)
            days = 7  # Varsayılan
            if context.args and len(context.args) > 0:
                try:
                    days = int(context.args[0])
                    if days < 1 or days > 365:
                        days = 7
                except ValueError:
                    days = 7
            
            # Performans raporu gönder
            success = await self.daily_reporter.send_performance_summary(days)
            
            if not success:
                await update.message.reply_text(
                    "❌ Performans raporu gönderilemedi.",
                    reply_markup=self.telegram_bot.main_keyboard
                )
            
        except Exception as e:
            self.logger.error(f"Performance komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Performans raporu alınamadı. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    async def realtime_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Realtime komutu - Real-time durum"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Real-time durum gönder
            success = await self.daily_reporter.send_real_time_status()
            
            if not success:
                await update.message.reply_text(
                    "❌ Real-time durum alınamadı.",
                    reply_markup=self.telegram_bot.main_keyboard
                )
            
        except Exception as e:
            self.logger.error(f"Realtime komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Real-time durum alınamadı. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    async def metrics_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Metrics komutu - Detaylı metrikler"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Real-time metrikler al
            metrics = self.monitoring.get_real_time_metrics()
            
            if not metrics:
                await update.message.reply_text(
                    "❌ Metrik veriler alınamadı.",
                    reply_markup=self.telegram_bot.main_keyboard
                )
                return
            
            # Formatla
            message = self._format_detailed_metrics(metrics)
            
            # Inline keyboard
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="refresh_metrics"),
                    InlineKeyboardButton("📊 Charts", callback_data="metrics_charts")
                ],
                [
                    InlineKeyboardButton("📈 Performance", callback_data="detailed_performance"),
                    InlineKeyboardButton("⚠️ Risk Metrics", callback_data="risk_metrics")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            self.logger.error(f"Metrics komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Metrik bilgisi alınamadı. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    async def trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Trades komutu - Son trade'ler"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Argüman kontrolü (kaç trade)
            limit = 10  # Varsayılan
            if context.args and len(context.args) > 0:
                try:
                    limit = int(context.args[0])
                    if limit < 1 or limit > 50:
                        limit = 10
                except ValueError:
                    limit = 10
            
            # Son trade'leri al
            trades = self.monitoring.db.get_trades(limit=limit)
            
            if not trades:
                await update.message.reply_text(
                    "📈 <b>No Recent Trades</b>\n\n"
                    "No trades found in the database.",
                    parse_mode='HTML',
                    reply_markup=self.telegram_bot.main_keyboard
                )
                return
            
            # Formatla
            message = self._format_trades_list(trades, limit)
            
            # Inline keyboard
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="refresh_trades"),
                    InlineKeyboardButton("📊 Analytics", callback_data="trade_analytics")
                ],
                [
                    InlineKeyboardButton("📈 Profitable", callback_data="profitable_trades"),
                    InlineKeyboardButton("📉 Losing", callback_data="losing_trades")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            self.logger.error(f"Trades komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Trade bilgisi alınamadı. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    async def signals_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Signals komutu - Son sinyaller"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Son sinyalleri al
            signals = self.monitoring.db.get_signals(limit=20)
            
            if not signals:
                await update.message.reply_text(
                    "🎯 <b>No Recent Signals</b>\n\n"
                    "No signals found in the database.",
                    parse_mode='HTML',
                    reply_markup=self.telegram_bot.main_keyboard
                )
                return
            
            # Formatla
            message = self._format_signals_list(signals)
            
            # Inline keyboard
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="refresh_signals"),
                    InlineKeyboardButton("📊 Model Stats", callback_data="model_stats")
                ],
                [
                    InlineKeyboardButton("🟢 Buy Signals", callback_data="buy_signals"),
                    InlineKeyboardButton("🔴 Sell Signals", callback_data="sell_signals")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            self.logger.error(f"Signals komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Sinyal bilgisi alınamadı. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    async def portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Portfolio komutu - Detaylı portföy analizi"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Portföy bilgilerini al
            portfolio = self.monitoring.db.get_portfolio()
            
            if not portfolio:
                await update.message.reply_text(
                    "💼 <b>Empty Portfolio</b>\n\n"
                    "No active positions found.",
                    parse_mode='HTML',
                    reply_markup=self.telegram_bot.main_keyboard
                )
                return
            
            # Formatla
            message = self._format_portfolio_analysis(portfolio)
            
            # Inline keyboard
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="refresh_portfolio"),
                    InlineKeyboardButton("📊 Analytics", callback_data="portfolio_analytics")
                ],
                [
                    InlineKeyboardButton("📈 Best Performers", callback_data="best_positions"),
                    InlineKeyboardButton("📉 Worst Performers", callback_data="worst_positions")
                ],
                [
                    InlineKeyboardButton("⚖️ Rebalance", callback_data="portfolio_rebalance")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            self.logger.error(f"Portfolio komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Portföy bilgisi alınamadı. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    async def risk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Risk komutu - Risk analizi"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Risk özetini al (risk manager'dan)
            # Bu gerçek uygulamada risk manager'dan gelecek
            risk_summary = {
                'current_drawdown': 0.05,
                'max_drawdown_limit': 0.15,
                'daily_pnl': -25.50,
                'daily_loss_limit': 0.05,
                'portfolio_risk': 0.08,
                'portfolio_risk_limit': 0.10,
                'active_alerts': 2,
                'emergency_stop': False
            }
            
            # Formatla
            message = self._format_risk_analysis(risk_summary)
            
            # Inline keyboard
            keyboard = [
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="refresh_risk"),
                    InlineKeyboardButton("⚠️ Alerts", callback_data="risk_alerts")
                ],
                [
                    InlineKeyboardButton("📊 Risk History", callback_data="risk_history"),
                    InlineKeyboardButton("⚙️ Risk Settings", callback_data="risk_settings")
                ],
                [
                    InlineKeyboardButton("🚨 Emergency Stop", callback_data="emergency_stop_risk")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            self.logger.error(f"Risk komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Risk bilgisi alınamadı. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    async def test_report_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Test report komutu - Test raporu gönder"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Test raporu gönder
            success = await self.daily_reporter.send_test_report()
            
            if success:
                await update.message.reply_text(
                    "✅ Test raporu başarıyla gönderildi!",
                    reply_markup=self.telegram_bot.main_keyboard
                )
            else:
                await update.message.reply_text(
                    "❌ Test raporu gönderilemedi.",
                    reply_markup=self.telegram_bot.main_keyboard
                )
            
        except Exception as e:
            self.logger.error(f"Test report komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Test raporu gönderilemedi. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    async def report_settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Report settings komutu - Rapor ayarları"""
        if not await self._check_authorization(update):
            return
        
        try:
            # Reporter durumunu al
            status = self.daily_reporter.get_status()
            
            # Formatla
            message = f"""
📊 <b>Report Settings</b>

<b>📅 Daily Reports:</b>
• Status: {'✅ Enabled' if status.get('enabled', False) else '❌ Disabled'}
• Schedule: {status.get('report_time', 'N/A')}
• Last Report: {status.get('last_report_date', 'Never')}

<b>🔄 Scheduler:</b>
• Running: {'✅ Yes' if status.get('scheduler_running', False) else '❌ No'}

<b>📱 Telegram:</b>
• Connected: {'✅ Yes' if status.get('telegram_connected', False) else '❌ No'}

<b>⚙️ Configuration:</b>
Use the buttons below to modify settings.
            """
            
            # Inline keyboard
            keyboard = [
                [
                    InlineKeyboardButton("✅ Enable Reports" if not status.get('enabled') else "❌ Disable Reports", 
                                       callback_data="toggle_reports"),
                    InlineKeyboardButton("⏰ Set Time", callback_data="set_report_time")
                ],
                [
                    InlineKeyboardButton("📤 Send Now", callback_data="send_report_now"),
                    InlineKeyboardButton("🧪 Test Report", callback_data="send_test_report")
                ],
                [
                    InlineKeyboardButton("🔄 Restart Scheduler", callback_data="restart_scheduler")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                message,
                parse_mode='HTML',
                reply_markup=reply_markup
            )
            
        except Exception as e:
            self.logger.error(f"Report settings komutu hatası: {e}")
            await update.message.reply_text(
                "❌ Rapor ayarları alınamadı. Lütfen tekrar deneyin.",
                reply_markup=self.telegram_bot.main_keyboard
            )
    
    # Helper Methods
    def _get_report_keyboard(self) -> InlineKeyboardMarkup:
        """Rapor için inline keyboard"""
        keyboard = [
            [
                InlineKeyboardButton("📊 Performance", callback_data="show_performance"),
                InlineKeyboardButton("📈 Trades", callback_data="show_trades")
            ],
            [
                InlineKeyboardButton("🎯 Signals", callback_data="show_signals"),
                InlineKeyboardButton("💼 Portfolio", callback_data="show_portfolio")
            ],
            [
                InlineKeyboardButton("⚠️ Risk Analysis", callback_data="show_risk")
            ]
        ]
        return InlineKeyboardMarkup(keyboard)
    
    def _format_detailed_metrics(self, metrics: Dict[str, Any]) -> str:
        """Detaylı metrikleri formatla"""
        try:
            lines = []
            lines.append("📊 <b>Detailed Metrics</b>")
            lines.append(f"🕐 {datetime.now().strftime('%H:%M:%S')}")
            lines.append("")
            
            # Ana metrikler
            lines.append("💰 <b>Financial Metrics:</b>")
            lines.append(f"• Current Capital: ${metrics.get('current_capital', 0):.2f}")
            lines.append(f"• Daily P&L: ${metrics.get('daily_pnl', 0):.2f}")
            lines.append(f"• Unrealized P&L: ${metrics.get('unrealized_pnl', 0):.2f}")
            lines.append("")
            
            # Trading metrikler
            lines.append("📈 <b>Trading Metrics:</b>")
            lines.append(f"• Active Positions: {metrics.get('active_positions', 0)}")
            lines.append(f"• Total Trades: {metrics.get('total_trades', 0)}")
            lines.append(f"• Win Rate: %{metrics.get('win_rate', 0)*100:.1f}")
            lines.append("")
            
            # Son aktiviteler
            last_trade = metrics.get('last_trade')
            if last_trade:
                lines.append("🔄 <b>Last Trade:</b>")
                lines.append(f"• {last_trade['symbol']} {last_trade['side']}")
                lines.append(f"• P&L: ${last_trade['pnl']:.2f}")
                lines.append("")
            
            last_signal = metrics.get('last_signal')
            if last_signal:
                lines.append("🎯 <b>Last Signal:</b>")
                lines.append(f"• {last_signal['model']}: {last_signal['symbol']}")
                lines.append(f"• Signal: {last_signal['signal']}")
                lines.append(f"• Confidence: %{last_signal['confidence']*100:.1f}")
                lines.append("")
            
            lines.append("🤖 *Real-time Trading Metrics*")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Metrics format error: {e}")
            return "❌ Metrik formatlanırken hata oluştu."
    
    def _format_trades_list(self, trades: List[Dict[str, Any]], limit: int) -> str:
        """Trade listesini formatla"""
        try:
            lines = []
            lines.append(f"📈 <b>Recent Trades (Last {limit})</b>")
            lines.append("")
            
            total_pnl = 0
            winning_trades = 0
            
            for i, trade in enumerate(trades[:limit], 1):
                pnl_emoji = "🟢" if trade['pnl'] > 0 else "🔴" if trade['pnl'] < 0 else "⚪"
                
                if trade['pnl'] > 0:
                    winning_trades += 1
                total_pnl += trade['pnl']
                
                lines.append(f"<b>{i}. {trade['symbol']}</b>")
                lines.append(f"  📊 {trade['side']} {trade['quantity']}")
                lines.append(f"  💵 ${trade['price']:.2f}")
                lines.append(f"  {pnl_emoji} ${trade['pnl']:.2f}")
                lines.append(f"  ⏰ {trade['timestamp'].strftime('%m/%d %H:%M')}")
                lines.append("")
            
            # Özet
            win_rate = (winning_trades / len(trades)) * 100 if trades else 0
            total_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
            
            lines.append(f"{total_emoji} <b>Summary:</b>")
            lines.append(f"• Total P&L: ${total_pnl:.2f}")
            lines.append(f"• Win Rate: %{win_rate:.1f}")
            lines.append(f"• Winning: {winning_trades}/{len(trades)}")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Trades format error: {e}")
            return "❌ Trade listesi formatlanırken hata oluştu."
    
    def _format_signals_list(self, signals: List[Dict[str, Any]]) -> str:
        """Sinyal listesini formatla"""
        try:
            lines = []
            lines.append("🎯 <b>Recent Signals</b>")
            lines.append("")
            
            # Model bazında grupla
            models = {}
            for signal in signals:
                model = signal['model']
                if model not in models:
                    models[model] = []
                models[model].append(signal)
            
            for model, model_signals in models.items():
                lines.append(f"<b>🤖 {model}:</b>")
                
                for signal in model_signals[:5]:  # Son 5 sinyal
                    signal_emoji = "🟢" if signal['signal'] == 'BUY' else "🔴" if signal['signal'] == 'SELL' else "⚪"
                    
                    lines.append(f"  {signal_emoji} {signal['symbol']} {signal['signal']}")
                    lines.append(f"    💪 %{signal['confidence']*100:.0f} confidence")
                    lines.append(f"    ⏰ {signal['timestamp'].strftime('%m/%d %H:%M')}")
                
                lines.append("")
            
            # Özet
            buy_signals = len([s for s in signals if s['signal'] == 'BUY'])
            sell_signals = len([s for s in signals if s['signal'] == 'SELL'])
            avg_confidence = sum(s['confidence'] for s in signals) / len(signals) if signals else 0
            
            lines.append("<b>📊 Summary:</b>")
            lines.append(f"• Total Signals: {len(signals)}")
            lines.append(f"• BUY: {buy_signals} | SELL: {sell_signals}")
            lines.append(f"• Avg Confidence: %{avg_confidence*100:.1f}")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Signals format error: {e}")
            return "❌ Sinyal listesi formatlanırken hata oluştu."
    
    def _format_portfolio_analysis(self, portfolio: List[Dict[str, Any]]) -> str:
        """Portföy analizini formatla"""
        try:
            lines = []
            lines.append("💼 <b>Portfolio Analysis</b>")
            lines.append("")
            
            total_value = 0
            total_unrealized = 0
            
            for pos in portfolio:
                value = pos['quantity'] * pos['current_price'] if pos['current_price'] else 0
                total_value += value
                total_unrealized += pos['unrealized_pnl']
                
                pnl_emoji = "🟢" if pos['unrealized_pnl'] > 0 else "🔴" if pos['unrealized_pnl'] < 0 else "⚪"
                pnl_percent = (pos['unrealized_pnl'] / (pos['quantity'] * pos['avg_price'])) * 100 if pos['avg_price'] > 0 else 0
                
                lines.append(f"<b>{pos['symbol']}</b>")
                lines.append(f"  💰 {pos['quantity']:.6f}")
                lines.append(f"  📊 Avg: ${pos['avg_price']:.2f}")
                lines.append(f"  📈 Current: ${pos['current_price']:.2f}")
                lines.append(f"  {pnl_emoji} ${pos['unrealized_pnl']:.2f} ({pnl_percent:+.2f}%)")
                lines.append(f"  💵 Value: ${value:.2f}")
                lines.append("")
            
            # Özet
            total_emoji = "🟢" if total_unrealized > 0 else "🔴" if total_unrealized < 0 else "⚪"
            
            lines.append(f"{total_emoji} <b>Portfolio Summary:</b>")
            lines.append(f"• Total Value: ${total_value:.2f}")
            lines.append(f"• Unrealized P&L: ${total_unrealized:.2f}")
            lines.append(f"• Positions: {len(portfolio)}")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Portfolio format error: {e}")
            return "❌ Portföy analizi formatlanırken hata oluştu."
    
    def _format_risk_analysis(self, risk_summary: Dict[str, Any]) -> str:
        """Risk analizini formatla"""
        try:
            lines = []
            lines.append("⚠️ <b>Risk Analysis</b>")
            lines.append("")
            
            # Drawdown analizi
            current_dd = risk_summary.get('current_drawdown', 0)
            max_dd_limit = risk_summary.get('max_drawdown_limit', 0.15)
            dd_emoji = "🟢" if current_dd < max_dd_limit * 0.5 else "🟡" if current_dd < max_dd_limit * 0.8 else "🔴"
            
            lines.append("📉 <b>Drawdown Analysis:</b>")
            lines.append(f"  {dd_emoji} Current: %{current_dd*100:.2f}")
            lines.append(f"  🎯 Limit: %{max_dd_limit*100:.2f}")
            lines.append("")
            
            # Günlük P&L analizi
            daily_pnl = risk_summary.get('daily_pnl', 0)
            daily_limit = risk_summary.get('daily_loss_limit', 0.05)
            daily_emoji = "🟢" if daily_pnl > 0 else "🟡" if abs(daily_pnl) < daily_limit * 0.8 else "🔴"
            
            lines.append("📊 <b>Daily P&L:</b>")
            lines.append(f"  {daily_emoji} Today: ${daily_pnl:.2f}")
            lines.append(f"  🎯 Loss Limit: %{daily_limit*100:.2f}")
            lines.append("")
            
            # Portföy riski
            portfolio_risk = risk_summary.get('portfolio_risk', 0)
            portfolio_limit = risk_summary.get('portfolio_risk_limit', 0.10)
            portfolio_emoji = "🟢" if portfolio_risk < portfolio_limit * 0.7 else "🟡" if portfolio_risk < portfolio_limit * 0.9 else "🔴"
            
            lines.append("💼 <b>Portfolio Risk:</b>")
            lines.append(f"  {portfolio_emoji} Current: %{portfolio_risk*100:.2f}")
            lines.append(f"  🎯 Limit: %{portfolio_limit*100:.2f}")
            lines.append("")
            
            # Alerts ve emergency
            active_alerts = risk_summary.get('active_alerts', 0)
            emergency_stop = risk_summary.get('emergency_stop', False)
            
            lines.append("🚨 <b>Risk Status:</b>")
            lines.append(f"  ⚠️ Active Alerts: {active_alerts}")
            lines.append(f"  🛑 Emergency Stop: {'🔴 ACTIVE' if emergency_stop else '🟢 INACTIVE'}")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Risk format error: {e}")
            return "❌ Risk analizi formatlanırken hata oluştu."
