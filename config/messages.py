"""
Словари локализации для веб-интерфейса.
Язык выбирается настройкой LANGUAGE в settings.py ('ru' или 'en').
"""

MESSAGES = {
    'ru': {
        # Статус хода (строка Turn в /game)
        'turn_prefix':        'Ход',
        'game_ready':         'Игра готова — нажмите следующий ход',
        'game_over':          'Игра окончена',
        'game_not_started':   'Игра не начата',

        # Состояния FSM (строка под ходом, если игра активна)
        'waiting_move':  'Бросок: {dice} → клетка {cell}',
        'awaiting_rule': 'Правило → клетка {cell}',
        'turn_done':     'Ход завершён',

        # Кнопка хода
        'btn_next':              '▶ Следующий ход',
        'btn_move_chip':         'Поставьте фишку на клетку {cell}',
        'btn_waiting':           'Ждите своего хода...',
        'btn_game_not_started':  'Ждите начала игры...',

        # Информация об игроке
        'playing_as': 'Вы',
        'chip_label': 'Фишка',
        'cell_label': 'клетка',
        'skip_label': 'пропуск',
        'you_label':  '[вы]',

        # Уведомления о правилах клеток
        'rule_skip_turn':    'Ход пропущен!',
        'rule_move_forward': 'Вперёд на {distance} клеток!',
        'rule_move_back':    'Назад на {distance} клеток!',
        'rule_finish':       'Финиш!',
        'rule_unknown':      'Правило: {effect}',

        # Страница /join
        'join_title':           'Присоединиться к игре',
        'join_name_label':      'Ваше имя',
        'join_name_placeholder':'Введите имя',
        'join_chip_label':      'Ваша фишка',
        'join_btn':             'Присоединиться →',
        'join_loading':         'Загрузка фишек...',
        'join_no_chips':        'Нет доступных фишек',
        'join_game_active':     'Игра уже идёт. Дождитесь окончания.',
        'join_error_name':      'Введите имя',
        'join_error_chip':      'Выберите фишку',
        'join_joining':         'Присоединяемся...',
        'join_error_network':   'Ошибка сети',
        'join_chip_taken':      'Фишка уже занята',

        # Страница /game: запрос имени
        'who_prompt':   'Кто вы?',
        'who_continue': 'Продолжить →',

        # Admin: веб-регистрация
        'reg_panel_title':   'Регистрация фишки',
        'reg_name_label':    'Название фишки',
        'reg_start_btn':     'Начать',
        'reg_cancel_btn':    'Отмена',
        'reg_drag_hint':     'Выделите фишку на изображении — снимок {num}/{total}',
        'reg_done':          'Фишка зарегистрирована!',
        'reg_error_name':    'Введите название',
    },

    'en': {
        'turn_prefix':      'Turn',
        'game_ready':       'Game ready — press Next Turn',
        'game_over':        'Game over',
        'game_not_started': 'Game not started',

        'waiting_move':  'Dice: {dice} → cell {cell}',
        'awaiting_rule': 'Rule → cell {cell}',
        'turn_done':     'Turn done',

        'btn_next':             '▶ Next Turn',
        'btn_move_chip':        'Place chip on cell {cell}',
        'btn_waiting':          'Waiting for your turn...',
        'btn_game_not_started': 'Waiting for game to start...',

        'playing_as': 'Playing as',
        'chip_label': 'Chip',
        'cell_label': 'cell',
        'skip_label': 'skip',
        'you_label':  '[you]',

        'rule_skip_turn':    'Turn skipped!',
        'rule_move_forward': 'Move forward {distance} cells!',
        'rule_move_back':    'Move back {distance} cells!',
        'rule_finish':       'Reached the finish!',
        'rule_unknown':      'Rule: {effect}',

        'join_title':           'Join Game',
        'join_name_label':      'Your name',
        'join_name_placeholder':'Enter your name',
        'join_chip_label':      'Your chip',
        'join_btn':             'Join →',
        'join_loading':         'Loading chips...',
        'join_no_chips':        'No chips available',
        'join_game_active':     'Game is already in progress. Wait for it to finish.',
        'join_error_name':      'Enter your name',
        'join_error_chip':      'Select a chip',
        'join_joining':         'Joining...',
        'join_error_network':   'Network error',
        'join_chip_taken':      'Chip already taken',

        'who_prompt':   'Who are you?',
        'who_continue': 'Continue →',

        'reg_panel_title': 'Register Chip',
        'reg_name_label':  'Chip name',
        'reg_start_btn':   'Start',
        'reg_cancel_btn':  'Cancel',
        'reg_drag_hint':   'Select chip on the stream — shot {num}/{total}',
        'reg_done':        'Chip registered!',
        'reg_error_name':  'Enter a name',
    },
}
