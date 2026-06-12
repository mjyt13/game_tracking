// /api/game/state
answer_state = {
    "status": "ok",
    "state": {
        "active": false,
        "turn_state": "IDLE",
        "expected_cell": 0,
        "rule_target": 0,
        "dice_roll": 0,
        "board_size": 12,
        "winner": null,
        "path_type": "linear",
        "current_turn": null,
        "ready_for_next": false,
        "players": [],
        "queue": [
            {
                "player": "player 1",
                "chip_name": "red",
                "chip_id": "b85dc0b0-c1d6-43af-b45f-6fe5163c6e01"
            },
            {
                "player": "player 2",
                "chip_name": "green",
                "chip_id": "3edda5a5-ea3a-4bc3-acbe-e75136298569"
            }
        ]
    }
}

// /api/game/events
answer_events = {
    "status": "ok",
    "events": [
        {
            "type": "GAME_START",
            "data": {
                "players": [
                    "player 1",
                    "player 2"
                ],
                "board_size": 12
            }
        },
        {
            "type": "TURN_START",
            "data": {
                "player": "player 1",
                "skipped": false,
                "dice_roll": 2,
                "current_cell": 0,
                "expected_cell": 2
            }
        }
    ],
    "game_state": {
        "active": true,
        "turn_state": "WAITING_MOVE",
        "expected_cell": 2,
        "rule_target": 0,
        "dice_roll": 2,
        "board_size": 12,
        "winner": null,
        "path_type": "linear",
        "current_turn": {
            "player": "player 1",
            "chip_id": "b85dc0b0-c1d6-43af-b45f-6fe5163c6e01",
            "chip_name": "red"
        },
        "ready_for_next": false,
        "players": [
            {
                "name": "player 1",
                "chip_id": "b85dc0b0-c1d6-43af-b45f-6fe5163c6e01",
                "chip_name": "red",
                "cell": 0,
                "skip_turns": 0
            },
            {
                "name": "player 2",
                "chip_id": "3edda5a5-ea3a-4bc3-acbe-e75136298569",
                "chip_name": "green",
                "cell": 0,
                "skip_turns": 0
            }
        ],
        "queue": [
            {
                "player": "player 1",
                "chip_name": "red",
                "chip_id": "b85dc0b0-c1d6-43af-b45f-6fe5163c6e01"
            },
            {
                "player": "player 2",
                "chip_name": "green",
                "chip_id": "3edda5a5-ea3a-4bc3-acbe-e75136298569"
            }
        ]
    }
}

// /api/chips
answer_chips = {
    "status": "ok",
    "chips": [
        {
            "id": "e86160a6-0e1f-4a6b-8145-e368d82fa363",
            "name": "black"
        },
        {
            "id": "ed7003ec-9329-44df-a519-9d3fbbdfaa5b",
            "name": "blue"
        },
        {
            "id": "7d97dfc6-3ca6-42d5-a634-b34d7671ef84",
            "name": "white"
        },
        {
            "id": "6119c317-91cf-4ba2-b6e7-bfa7b197465c",
            "name": "white2"
        },
        {
            "id": "cf81bfbc-d28e-40e9-a946-cce2dcf7cd55",
            "name": "blue2"
        }
    ]
}