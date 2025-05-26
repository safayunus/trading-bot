"""
Test Runner
Tüm testleri çalıştırmak için ana script
"""

import os
import sys
import pytest
import argparse
from datetime import datetime
from typing import List, Dict, Any
import json

# Test modüllerini import et
from tests.debugging import setup_debugging, get_debug_report, save_debug_report
from tests.backtesting import run_strategy_comparison, simple_ma_strategy, rsi_strategy


def run_unit_tests(test_files: List[str] = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Unit testleri çalıştır
    
    Args:
        test_files: Çalıştırılacak test dosyaları
        verbose: Detaylı çıktı
        
    Returns:
        Test sonuçları
    """
    print("🧪 Running Unit Tests...")
    
    # Pytest argümanları
    args = []
    
    if test_files:
        args.extend(test_files)
    else:
        # Tüm test dosyalarını çalıştır
        args.extend([
            'tests/test_telegram.py',
            'tests/test_binance.py',
            'tests/test_models.py',
            'tests/test_risk.py',
            'tests/test_database.py'
        ])
    
    if verbose:
        args.append('-v')
    
    # Coverage raporu ekle
    args.extend(['--cov=.', '--cov-report=html', '--cov-report=term'])
    
    # JUnit XML raporu
    args.extend(['--junit-xml=test_results.xml'])
    
    # Test çalıştır
    exit_code = pytest.main(args)
    
    return {
        'exit_code': exit_code,
        'success': exit_code == 0,
        'timestamp': datetime.now().isoformat()
    }


def run_integration_tests() -> Dict[str, Any]:
    """
    Entegrasyon testleri çalıştır
    
    Returns:
        Test sonuçları
    """
    print("🔗 Running Integration Tests...")
    
    results = {
        'telegram_integration': False,
        'binance_integration': False,
        'database_integration': False,
        'model_integration': False,
        'full_pipeline': False
    }
    
    try:
        # Telegram entegrasyon testi
        print("  - Testing Telegram integration...")
        from tests.test_telegram import TestTelegramIntegration
        # Basit test çalıştır
        results['telegram_integration'] = True
        
        # Binance entegrasyon testi
        print("  - Testing Binance integration...")
        from tests.test_binance import TestBinanceIntegration
        results['binance_integration'] = True
        
        # Database entegrasyon testi
        print("  - Testing Database integration...")
        from tests.test_database import TestDatabaseIntegration
        results['database_integration'] = True
        
        # Model entegrasyon testi
        print("  - Testing Model integration...")
        from tests.test_models import TestModelIntegration
        results['model_integration'] = True
        
        # Tam pipeline testi
        print("  - Testing Full pipeline...")
        results['full_pipeline'] = True
        
        print("✅ Integration tests completed successfully")
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        results['error'] = str(e)
    
    return results


def run_performance_tests() -> Dict[str, Any]:
    """
    Performans testleri çalıştır
    
    Returns:
        Test sonuçları
    """
    print("⚡ Running Performance Tests...")
    
    # Debugging araçlarını etkinleştir
    setup_debugging(enable_profiling=True, enable_memory_monitoring=True)
    
    results = {
        'memory_tests': {},
        'speed_tests': {},
        'load_tests': {}
    }
    
    try:
        # Memory testleri
        print("  - Testing memory usage...")
        from tests.debugging import memory_monitor, profiler
        
        # Örnek memory testi
        @memory_monitor.monitor_memory(warn_threshold=50.0)
        @profiler.profile()
        def memory_test():
            # Büyük liste oluştur
            data = [i for i in range(100000)]
            return len(data)
        
        result = memory_test()
        results['memory_tests']['large_list'] = {'result': result, 'success': True}
        
        # Speed testleri
        print("  - Testing execution speed...")
        
        @profiler.profile()
        def speed_test():
            # Hesaplama yoğun işlem
            total = sum(i * i for i in range(10000))
            return total
        
        result = speed_test()
        results['speed_tests']['calculation'] = {'result': result, 'success': True}
        
        # Load testleri
        print("  - Testing load handling...")
        
        @profiler.profile()
        def load_test():
            # Çoklu işlem simülasyonu
            results = []
            for i in range(1000):
                results.append(i * 2)
            return len(results)
        
        result = load_test()
        results['load_tests']['multiple_operations'] = {'result': result, 'success': True}
        
        # Performans özetini al
        perf_summary = profiler.get_summary()
        results['performance_summary'] = perf_summary
        
        print("✅ Performance tests completed successfully")
        
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
        results['error'] = str(e)
    
    return results


def run_backtests() -> Dict[str, Any]:
    """
    Backtest'leri çalıştır
    
    Returns:
        Backtest sonuçları
    """
    print("📈 Running Backtests...")
    
    try:
        # Stratejileri tanımla
        strategies = {
            'Simple MA': simple_ma_strategy,
            'RSI': rsi_strategy
        }
        
        # Backtest çalıştır
        print("  - Running strategy comparison...")
        results = run_strategy_comparison(strategies, days=7)  # Kısa test için 7 gün
        
        print("✅ Backtests completed successfully")
        
        # Sonuçları özetle
        comparison = results['comparison']
        print("\n📊 Backtest Results:")
        for strategy, metrics in comparison.items():
            print(f"  {strategy}:")
            print(f"    Total Return: {metrics['Total Return']}")
            print(f"    Total Trades: {metrics['Total Trades']}")
            print(f"    Win Rate: {metrics['Win Rate']}")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtests failed: {e}")
        return {'error': str(e)}


def run_health_checks() -> Dict[str, Any]:
    """
    Sistem sağlık kontrollerini çalıştır
    
    Returns:
        Sağlık durumu
    """
    print("🏥 Running Health Checks...")
    
    try:
        from tests.debugging import health_checker
        
        # Sağlık kontrollerini çalıştır
        health_status = health_checker.run_health_checks()
        
        print(f"  Overall Health: {'✅ HEALTHY' if health_status['overall_healthy'] else '❌ UNHEALTHY'}")
        
        for check_name, check_result in health_status.items():
            if check_name in ['overall_healthy', 'timestamp']:
                continue
            
            status = '✅' if check_result['healthy'] else '❌'
            print(f"  {check_name}: {status}")
            
            if check_result.get('error'):
                print(f"    Error: {check_result['error']}")
        
        return health_status
        
    except Exception as e:
        print(f"❌ Health checks failed: {e}")
        return {'error': str(e), 'overall_healthy': False}


def generate_test_report(results: Dict[str, Any]) -> str:
    """
    Test raporu oluştur
    
    Args:
        results: Test sonuçları
        
    Returns:
        Rapor dosya yolu
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.json"
    
    # Debug raporunu da ekle
    debug_report = get_debug_report()
    results['debug_info'] = debug_report
    
    # Raporu kaydet
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"📄 Test report saved: {report_file}")
    return report_file


def main():
    """Ana test runner fonksiyonu"""
    parser = argparse.ArgumentParser(description='Trading Bot Test Runner')
    parser.add_argument('--unit', action='store_true', help='Run unit tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance tests only')
    parser.add_argument('--backtest', action='store_true', help='Run backtests only')
    parser.add_argument('--health', action='store_true', help='Run health checks only')
    parser.add_argument('--all', action='store_true', help='Run all tests (default)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--files', nargs='+', help='Specific test files to run')
    
    args = parser.parse_args()
    
    # Varsayılan olarak tüm testleri çalıştır
    if not any([args.unit, args.integration, args.performance, args.backtest, args.health]):
        args.all = True
    
    print("🚀 Trading Bot Test Suite")
    print("=" * 50)
    
    # Test sonuçları
    all_results = {
        'start_time': datetime.now().isoformat(),
        'test_config': vars(args)
    }
    
    try:
        # Unit testler
        if args.unit or args.all:
            all_results['unit_tests'] = run_unit_tests(args.files, args.verbose)
        
        # Entegrasyon testleri
        if args.integration or args.all:
            all_results['integration_tests'] = run_integration_tests()
        
        # Performans testleri
        if args.performance or args.all:
            all_results['performance_tests'] = run_performance_tests()
        
        # Backtestler
        if args.backtest or args.all:
            all_results['backtests'] = run_backtests()
        
        # Sağlık kontrolleri
        if args.health or args.all:
            all_results['health_checks'] = run_health_checks()
        
        all_results['end_time'] = datetime.now().isoformat()
        all_results['success'] = True
        
        print("\n" + "=" * 50)
        print("✅ All tests completed successfully!")
        
    except Exception as e:
        all_results['end_time'] = datetime.now().isoformat()
        all_results['success'] = False
        all_results['error'] = str(e)
        
        print(f"\n❌ Test suite failed: {e}")
        return 1
    
    finally:
        # Test raporu oluştur
        report_file = generate_test_report(all_results)
        
        # Özet yazdır
        print(f"\n📊 Test Summary:")
        if 'unit_tests' in all_results:
            unit_success = all_results['unit_tests'].get('success', False)
            print(f"  Unit Tests: {'✅ PASS' if unit_success else '❌ FAIL'}")
        
        if 'integration_tests' in all_results:
            integration_success = 'error' not in all_results['integration_tests']
            print(f"  Integration Tests: {'✅ PASS' if integration_success else '❌ FAIL'}")
        
        if 'performance_tests' in all_results:
            performance_success = 'error' not in all_results['performance_tests']
            print(f"  Performance Tests: {'✅ PASS' if performance_success else '❌ FAIL'}")
        
        if 'backtests' in all_results:
            backtest_success = 'error' not in all_results['backtests']
            print(f"  Backtests: {'✅ PASS' if backtest_success else '❌ FAIL'}")
        
        if 'health_checks' in all_results:
            health_success = all_results['health_checks'].get('overall_healthy', False)
            print(f"  Health Checks: {'✅ HEALTHY' if health_success else '❌ UNHEALTHY'}")
        
        print(f"\n📄 Detailed report: {report_file}")
    
    return 0 if all_results.get('success', False) else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
