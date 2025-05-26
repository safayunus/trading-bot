"""
Advanced Monitoring and Reporting System
Real-time monitoring, performance tracking ve otomatik raporlama
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json

from .database import AdvancedDatabaseManager

@dataclass
class DailyReport:
    """Günlük rapor"""
    date: str
    trades_summary: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    portfolio_status: Dict[str, Any]
    signals_summary: Dict[str, Any]
    risk_events: List[Dict[str, Any]]
    model_performance: Dict[str, Any]
    recommendations: List[str]

class MonitoringSystem:
    """Gelişmiş monitoring sistemi"""
    
    def __init__(self, db_manager: AdvancedDatabaseManager):
        """
        Monitoring system başlatıcı
        
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_cache = {}
        self.last_report_date = None
        
    def generate_daily_report(self) -> DailyReport:
        """
        Günlük rapor oluştur
        
        Returns:
            DailyReport: Günlük rapor
        """
        try:
            today = datetime.now().date()
            yesterday = today - timedelta(days=1)
            
            # Trades summary
            trades_summary = self._get_trades_summary(today)
            
            # Performance metrics
            performance_metrics = self._get_performance_metrics(today)
            
            # Portfolio status
            portfolio_status = self._get_portfolio_status()
            
            # Signals summary
            signals_summary = self._get_signals_summary(today)
            
            # Risk events
            risk_events = self._get_risk_events(today)
            
            # Model performance
            model_performance = self._get_model_performance_summary()
            
            # Recommendations
            recommendations = self._generate_recommendations(
                trades_summary, performance_metrics, portfolio_status
            )
            
            report = DailyReport(
                date=today.isoformat(),
                trades_summary=trades_summary,
                performance_metrics=performance_metrics,
                portfolio_status=portfolio_status,
                signals_summary=signals_summary,
                risk_events=risk_events,
                model_performance=model_performance,
                recommendations=recommendations
            )
            
            # Raporu veritabanına kaydet
            self._save_report_to_db(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Daily report generation error: {e}")
            return self._create_empty_report()
    
    def _get_trades_summary(self, date: datetime.date) -> Dict[str, Any]:
        """Günlük trade özeti"""
        try:
            # Bugünkü trades
            trades = self.db.get_trades(days=1)
            today_trades = [t for t in trades if t['timestamp'].date() == date]
            
            if not today_trades:
                return {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'total_pnl': 0.0,
                    'avg_trade_pnl': 0.0,
                    'best_trade': 0.0,
                    'worst_trade': 0.0,
                    'total_volume': 0.0,
                    'symbols_traded': []
                }
            
            total_trades = len(today_trades)
            winning_trades = len([t for t in today_trades if t['pnl'] > 0])
            losing_trades = len([t for t in today_trades if t['pnl'] < 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t['pnl'] for t in today_trades)
            avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
            
            pnls = [t['pnl'] for t in today_trades]
            best_trade = max(pnls) if pnls else 0
            worst_trade = min(pnls) if pnls else 0
            
            total_volume = sum(t['quantity'] * t['price'] for t in today_trades)
            symbols_traded = list(set(t['symbol'] for t in today_trades))
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_trade_pnl': avg_trade_pnl,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'total_volume': total_volume,
                'symbols_traded': symbols_traded
            }
            
        except Exception as e:
            self.logger.error(f"Trades summary error: {e}")
            return {}
    
    def _get_performance_metrics(self, date: datetime.date) -> Dict[str, Any]:
        """Performance metrikleri"""
        try:
            # 30 günlük performans özeti
            performance_summary = self.db.get_performance_summary(days=30)
            
            if not performance_summary:
                return {}
            
            # Bugünkü performans
            today_performance = self.db.get_performance_summary(days=1)
            
            return {
                'daily_return': today_performance.get('total_return', 0.0),
                'total_return_30d': performance_summary.get('total_return', 0.0),
                'sharpe_ratio': performance_summary.get('sharpe_ratio', 0.0),
                'max_drawdown': performance_summary.get('max_drawdown', 0.0),
                'win_rate_30d': performance_summary.get('win_rate', 0.0),
                'profit_factor': performance_summary.get('profit_factor', 0.0),
                'current_capital': performance_summary.get('current_capital', 0.0),
                'unrealized_pnl': performance_summary.get('unrealized_pnl', 0.0),
                'total_trades_30d': performance_summary.get('total_trades', 0),
                'volatility': self._calculate_volatility()
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics error: {e}")
            return {}
    
    def _get_portfolio_status(self) -> Dict[str, Any]:
        """Portfolio durumu"""
        try:
            portfolio = self.db.get_portfolio()
            
            if not portfolio:
                return {
                    'total_positions': 0,
                    'total_value': 0.0,
                    'total_unrealized_pnl': 0.0,
                    'positions': []
                }
            
            total_positions = len(portfolio)
            total_value = sum(pos['quantity'] * pos['current_price'] for pos in portfolio if pos['current_price'])
            total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in portfolio)
            
            # En iyi ve en kötü performans gösteren pozisyonlar
            best_position = max(portfolio, key=lambda x: x['unrealized_pnl']) if portfolio else None
            worst_position = min(portfolio, key=lambda x: x['unrealized_pnl']) if portfolio else None
            
            return {
                'total_positions': total_positions,
                'total_value': total_value,
                'total_unrealized_pnl': total_unrealized_pnl,
                'best_position': {
                    'symbol': best_position['symbol'],
                    'unrealized_pnl': best_position['unrealized_pnl']
                } if best_position else None,
                'worst_position': {
                    'symbol': worst_position['symbol'],
                    'unrealized_pnl': worst_position['unrealized_pnl']
                } if worst_position else None,
                'positions': [
                    {
                        'symbol': pos['symbol'],
                        'quantity': pos['quantity'],
                        'avg_price': pos['avg_price'],
                        'current_price': pos['current_price'],
                        'unrealized_pnl': pos['unrealized_pnl'],
                        'pnl_percent': (pos['unrealized_pnl'] / (pos['quantity'] * pos['avg_price'])) * 100 if pos['avg_price'] > 0 else 0
                    }
                    for pos in portfolio
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio status error: {e}")
            return {}
    
    def _get_signals_summary(self, date: datetime.date) -> Dict[str, Any]:
        """Sinyal özeti"""
        try:
            # Bugünkü sinyaller
            signals = self.db.get_signals(days=1)
            today_signals = [s for s in signals if s['timestamp'].date() == date]
            
            if not today_signals:
                return {
                    'total_signals': 0,
                    'buy_signals': 0,
                    'sell_signals': 0,
                    'hold_signals': 0,
                    'avg_confidence': 0.0,
                    'models_active': [],
                    'symbols_analyzed': []
                }
            
            total_signals = len(today_signals)
            buy_signals = len([s for s in today_signals if s['signal'] == 'BUY'])
            sell_signals = len([s for s in today_signals if s['signal'] == 'SELL'])
            hold_signals = len([s for s in today_signals if s['signal'] == 'HOLD'])
            
            avg_confidence = np.mean([s['confidence'] for s in today_signals])
            
            models_active = list(set(s['model'] for s in today_signals))
            symbols_analyzed = list(set(s['symbol'] for s in today_signals))
            
            # Model bazında sinyal dağılımı
            model_signals = {}
            for model in models_active:
                model_sigs = [s for s in today_signals if s['model'] == model]
                model_signals[model] = {
                    'total': len(model_sigs),
                    'buy': len([s for s in model_sigs if s['signal'] == 'BUY']),
                    'sell': len([s for s in model_sigs if s['signal'] == 'SELL']),
                    'hold': len([s for s in model_sigs if s['signal'] == 'HOLD']),
                    'avg_confidence': np.mean([s['confidence'] for s in model_sigs])
                }
            
            return {
                'total_signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'hold_signals': hold_signals,
                'avg_confidence': avg_confidence,
                'models_active': models_active,
                'symbols_analyzed': symbols_analyzed,
                'model_breakdown': model_signals
            }
            
        except Exception as e:
            self.logger.error(f"Signals summary error: {e}")
            return {}
    
    def _get_risk_events(self, date: datetime.date) -> List[Dict[str, Any]]:
        """Risk olayları"""
        try:
            # Bugünkü risk olayları (logs tablosundan)
            logs = self.db.get_logs(level='WARNING', days=1)
            logs.extend(self.db.get_logs(level='ERROR', days=1))
            logs.extend(self.db.get_logs(level='CRITICAL', days=1))
            
            today_logs = [log for log in logs if log['timestamp'].date() == date]
            
            risk_events = []
            for log in today_logs:
                if any(keyword in log['message'].lower() for keyword in ['risk', 'loss', 'drawdown', 'emergency']):
                    risk_events.append({
                        'timestamp': log['timestamp'].isoformat(),
                        'level': log['level'],
                        'message': log['message'],
                        'module': log['module']
                    })
            
            return risk_events
            
        except Exception as e:
            self.logger.error(f"Risk events error: {e}")
            return []
    
    def _get_model_performance_summary(self) -> Dict[str, Any]:
        """Model performans özeti"""
        try:
            # Son 7 günün sinyalleri
            signals = self.db.get_signals(days=7)
            
            if not signals:
                return {}
            
            # Model bazında performans
            models = list(set(s['model'] for s in signals))
            model_performance = {}
            
            for model in models:
                model_signals = [s for s in signals if s['model'] == model]
                
                total_signals = len(model_signals)
                avg_confidence = np.mean([s['confidence'] for s in model_signals])
                
                # Sinyal dağılımı
                buy_signals = len([s for s in model_signals if s['signal'] == 'BUY'])
                sell_signals = len([s for s in model_signals if s['signal'] == 'SELL'])
                hold_signals = len([s for s in model_signals if s['signal'] == 'HOLD'])
                
                model_performance[model] = {
                    'total_signals': total_signals,
                    'avg_confidence': avg_confidence,
                    'buy_signals': buy_signals,
                    'sell_signals': sell_signals,
                    'hold_signals': hold_signals,
                    'signal_distribution': {
                        'buy_pct': (buy_signals / total_signals) * 100 if total_signals > 0 else 0,
                        'sell_pct': (sell_signals / total_signals) * 100 if total_signals > 0 else 0,
                        'hold_pct': (hold_signals / total_signals) * 100 if total_signals > 0 else 0
                    }
                }
            
            return model_performance
            
        except Exception as e:
            self.logger.error(f"Model performance summary error: {e}")
            return {}
    
    def _generate_recommendations(self, trades_summary: Dict[str, Any], 
                                performance_metrics: Dict[str, Any],
                                portfolio_status: Dict[str, Any]) -> List[str]:
        """Öneriler oluştur"""
        try:
            recommendations = []
            
            # Trade performance bazlı öneriler
            if trades_summary.get('win_rate', 0) < 0.5:
                recommendations.append("⚠️ Win rate düşük (%{:.1f}). Strateji gözden geçirilmeli.".format(
                    trades_summary.get('win_rate', 0) * 100
                ))
            
            if trades_summary.get('total_pnl', 0) < 0:
                recommendations.append("📉 Günlük P&L negatif. Risk yönetimi kontrol edilmeli.")
            
            # Drawdown kontrolü
            max_drawdown = performance_metrics.get('max_drawdown', 0)
            if max_drawdown > 0.1:  # %10'dan fazla
                recommendations.append("🚨 Max drawdown yüksek (%{:.1f}). Pozisyon boyutları azaltılmalı.".format(
                    max_drawdown * 100
                ))
            
            # Portfolio diversification
            total_positions = portfolio_status.get('total_positions', 0)
            if total_positions > 10:
                recommendations.append("📊 Çok fazla pozisyon ({} adet). Portföy sadeleştirilmeli.".format(total_positions))
            elif total_positions < 3 and total_positions > 0:
                recommendations.append("📈 Az pozisyon ({} adet). Diversifikasyon artırılabilir.".format(total_positions))
            
            # Unrealized P&L kontrolü
            unrealized_pnl = portfolio_status.get('total_unrealized_pnl', 0)
            if unrealized_pnl < -1000:  # $1000'dan fazla kayıp
                recommendations.append("💰 Büyük unrealized loss (${:.2f}). Stop loss'lar gözden geçirilmeli.".format(unrealized_pnl))
            
            # Sharpe ratio kontrolü
            sharpe_ratio = performance_metrics.get('sharpe_ratio', 0)
            if sharpe_ratio < 1.0:
                recommendations.append("📊 Sharpe ratio düşük ({:.2f}). Risk-adjusted return iyileştirilebilir.".format(sharpe_ratio))
            
            # Pozitif durumlar için övgü
            if trades_summary.get('win_rate', 0) > 0.7:
                recommendations.append("✅ Mükemmel win rate (%{:.1f})! Mevcut strateji korunmalı.".format(
                    trades_summary.get('win_rate', 0) * 100
                ))
            
            if performance_metrics.get('daily_return', 0) > 0.02:  # %2'den fazla günlük return
                recommendations.append("🚀 Harika günlük performans! Risk yönetimi ile devam edilmeli.")
            
            if not recommendations:
                recommendations.append("✅ Tüm metrikler normal aralıkta. Mevcut strateji ile devam edilebilir.")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendations generation error: {e}")
            return ["❌ Öneriler oluşturulurken hata oluştu."]
    
    def _calculate_volatility(self, days: int = 30) -> float:
        """Volatilite hesapla"""
        try:
            # Son X günün günlük return'lerini al
            performance_summary = self.db.get_performance_summary(days=days)
            
            # Basit volatilite hesaplaması
            # Gerçek uygulamada günlük return'lerin std'si alınmalı
            return performance_summary.get('max_drawdown', 0.0) * 2  # Approximation
            
        except Exception as e:
            self.logger.error(f"Volatility calculation error: {e}")
            return 0.0
    
    def _save_report_to_db(self, report: DailyReport):
        """Raporu veritabanına kaydet"""
        try:
            # Report'u JSON olarak settings tablosuna kaydet
            report_data = {
                'date': report.date,
                'trades_summary': report.trades_summary,
                'performance_metrics': report.performance_metrics,
                'portfolio_status': report.portfolio_status,
                'signals_summary': report.signals_summary,
                'risk_events': report.risk_events,
                'model_performance': report.model_performance,
                'recommendations': report.recommendations
            }
            
            self.db.set_setting(
                f'daily_report_{report.date}',
                json.dumps(report_data),
                f'Daily report for {report.date}'
            )
            
            # Son rapor tarihini güncelle
            self.db.set_setting('last_report_date', report.date)
            
        except Exception as e:
            self.logger.error(f"Report save error: {e}")
    
    def _create_empty_report(self) -> DailyReport:
        """Boş rapor oluştur"""
        today = datetime.now().date()
        
        return DailyReport(
            date=today.isoformat(),
            trades_summary={},
            performance_metrics={},
            portfolio_status={},
            signals_summary={},
            risk_events=[],
            model_performance={},
            recommendations=["❌ Rapor oluşturulurken hata oluştu."]
        )
    
    def format_report_for_telegram(self, report: DailyReport) -> str:
        """
        Telegram için rapor formatla
        
        Args:
            report: Günlük rapor
            
        Returns:
            str: Formatlanmış rapor metni
        """
        try:
            lines = []
            lines.append(f"📊 **Günlük Trading Raporu**")
            lines.append(f"📅 Tarih: {report.date}")
            lines.append("")
            
            # Trades Summary
            trades = report.trades_summary
            if trades:
                lines.append("💼 **Trade Özeti:**")
                lines.append(f"• Toplam Trade: {trades.get('total_trades', 0)}")
                lines.append(f"• Kazanan: {trades.get('winning_trades', 0)} | Kaybeden: {trades.get('losing_trades', 0)}")
                lines.append(f"• Win Rate: %{trades.get('win_rate', 0)*100:.1f}")
                lines.append(f"• Günlük P&L: ${trades.get('total_pnl', 0):.2f}")
                lines.append(f"• En İyi Trade: ${trades.get('best_trade', 0):.2f}")
                lines.append(f"• En Kötü Trade: ${trades.get('worst_trade', 0):.2f}")
                lines.append("")
            
            # Performance Metrics
            perf = report.performance_metrics
            if perf:
                lines.append("📈 **Performans Metrikleri:**")
                lines.append(f"• Günlük Return: %{perf.get('daily_return', 0)*100:.2f}")
                lines.append(f"• 30 Gün Return: %{perf.get('total_return_30d', 0)*100:.2f}")
                lines.append(f"• Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
                lines.append(f"• Max Drawdown: %{perf.get('max_drawdown', 0)*100:.2f}")
                lines.append(f"• Mevcut Sermaye: ${perf.get('current_capital', 0):.2f}")
                lines.append("")
            
            # Portfolio Status
            portfolio = report.portfolio_status
            if portfolio:
                lines.append("💰 **Portföy Durumu:**")
                lines.append(f"• Aktif Pozisyon: {portfolio.get('total_positions', 0)}")
                lines.append(f"• Toplam Değer: ${portfolio.get('total_value', 0):.2f}")
                lines.append(f"• Unrealized P&L: ${portfolio.get('total_unrealized_pnl', 0):.2f}")
                
                if portfolio.get('best_position'):
                    best = portfolio['best_position']
                    lines.append(f"• En İyi: {best['symbol']} (${best['unrealized_pnl']:.2f})")
                
                if portfolio.get('worst_position'):
                    worst = portfolio['worst_position']
                    lines.append(f"• En Kötü: {worst['symbol']} (${worst['unrealized_pnl']:.2f})")
                lines.append("")
            
            # Signals Summary
            signals = report.signals_summary
            if signals:
                lines.append("🎯 **Sinyal Özeti:**")
                lines.append(f"• Toplam Sinyal: {signals.get('total_signals', 0)}")
                lines.append(f"• BUY: {signals.get('buy_signals', 0)} | SELL: {signals.get('sell_signals', 0)} | HOLD: {signals.get('hold_signals', 0)}")
                lines.append(f"• Ortalama Güven: %{signals.get('avg_confidence', 0)*100:.1f}")
                lines.append(f"• Aktif Modeller: {len(signals.get('models_active', []))}")
                lines.append("")
            
            # Risk Events
            if report.risk_events:
                lines.append("⚠️ **Risk Olayları:**")
                for event in report.risk_events[:3]:  # İlk 3 olay
                    lines.append(f"• {event['level']}: {event['message'][:50]}...")
                lines.append("")
            
            # Recommendations
            if report.recommendations:
                lines.append("💡 **Öneriler:**")
                for rec in report.recommendations[:5]:  # İlk 5 öneri
                    lines.append(f"• {rec}")
                lines.append("")
            
            lines.append("🤖 *Otomatik rapor - Trading Bot*")
            
            return "\n".join(lines)
            
        except Exception as e:
            self.logger.error(f"Telegram format error: {e}")
            return f"❌ Rapor formatlanırken hata oluştu: {str(e)}"
    
    async def should_send_daily_report(self) -> bool:
        """
        Günlük rapor gönderilmeli mi kontrol et
        
        Returns:
            bool: Rapor gönderilmeli mi
        """
        try:
            # Telegram reports enabled mi?
            reports_enabled = self.db.get_setting('telegram_reports_enabled', True)
            if not reports_enabled:
                return False
            
            # Son rapor tarihi
            last_report_date = self.db.get_setting('last_report_date')
            today = datetime.now().date()
            
            if last_report_date:
                last_date = datetime.strptime(last_report_date, '%Y-%m-%d').date()
                return today > last_date
            
            return True
            
        except Exception as e:
            self.logger.error(f"Should send report check error: {e}")
            return False
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """
        Real-time metrikler
        
        Returns:
            Dict: Anlık metrikler
        """
        try:
            # Son 24 saatin özeti
            performance = self.db.get_performance_summary(days=1)
            portfolio = self.db.get_portfolio()
            
            # Son trade
            recent_trades = self.db.get_trades(limit=1)
            last_trade = recent_trades[0] if recent_trades else None
            
            # Son sinyal
            recent_signals = self.db.get_signals(limit=1)
            last_signal = recent_signals[0] if recent_signals else None
            
            return {
                'timestamp': datetime.now().isoformat(),
                'current_capital': performance.get('current_capital', 0),
                'daily_pnl': performance.get('total_pnl', 0),
                'unrealized_pnl': performance.get('unrealized_pnl', 0),
                'active_positions': len(portfolio),
                'last_trade': {
                    'symbol': last_trade['symbol'],
                    'side': last_trade['side'],
                    'pnl': last_trade['pnl'],
                    'timestamp': last_trade['timestamp'].isoformat()
                } if last_trade else None,
                'last_signal': {
                    'model': last_signal['model'],
                    'symbol': last_signal['symbol'],
                    'signal': last_signal['signal'],
                    'confidence': last_signal['confidence'],
                    'timestamp': last_signal['timestamp'].isoformat()
                } if last_signal else None,
                'win_rate': performance.get('win_rate', 0),
                'total_trades': performance.get('total_trades', 0)
            }
            
        except Exception as e:
            self.logger.error(f"Real-time metrics error: {e}")
            return {}
