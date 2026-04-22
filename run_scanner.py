"""
Главный файл запуска системы сканирования фишек
Использует новую модульную архитектуру
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

from core.game_scanner import main

if __name__ == "__main__":
    main()

