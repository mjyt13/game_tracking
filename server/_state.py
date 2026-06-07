import queue


class _State:
    events: list = []
    game_state: dict = {}
    stream_frame: bytes = b""
    command_queue: queue.Queue = queue.Queue()   # "next" | "start"
    available_chips: list = []                   # [{id, name}] — незанятые фишки
    all_chips: list = []                         # [{id, name}] — все зарегистрированные
    chip_images: dict = {}                       # chip_id -> JPEG bytes (миниатюра)
    join_queue: queue.Queue = queue.Queue()      # {name: str, chip_id: str}
    reg_queue: queue.Queue = queue.Queue()       # {action: "start"|"shot"|"cancel", ...}
    admin_queue: queue.Queue = queue.Queue()     # {action: "delete_chip"|"remove_player", ...}
    reg_status: dict = {"active": False, "shot_num": 0, "total": 5, "name": "", "done": False}
