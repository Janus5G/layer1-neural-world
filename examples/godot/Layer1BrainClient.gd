# Godot 4.x. Attach to a Node3D and assign target_path in the Inspector.
# The game engine owns physics and visuals; the local bridge returns only an action.
extends Node3D

@export var server_url: String = "ws://127.0.0.1:9090/engine/ws"
@export var target_path: NodePath
@export var agent_id: String = "godot-agent-1"
@export var enable_learning: bool = false
@export var speed: float = 3.0
@export var step_seconds: float = 0.3

var ws := WebSocketPeer.new()
var target: Node3D
var registered := false
var waiting := false
var next_step := 0.0
var last_reward := 0.0
var previous_distance := 0.0

func _ready() -> void:
    target = get_node_or_null(target_path) as Node3D
    if target == null:
        push_error("Assign a target Node3D in the inspector")
        set_process(false)
        return
    previous_distance = global_position.distance_to(target.global_position)
    var result := ws.connect_to_url(server_url)
    if result != OK:
        push_error("Unable to connect to Engine Bridge: %s" % result)
        set_process(false)

func _process(delta: float) -> void:
    ws.poll()
    var connection_state := ws.get_ready_state()
    if connection_state == WebSocketPeer.STATE_CLOSED:
        registered = false
        return
    if connection_state != WebSocketPeer.STATE_OPEN:
        return
    if not registered and not waiting:
        _send({"type": "create", "agentId": agent_id, "inputDim": 6,
            "actions": ["north", "south", "east", "west", "idle"], "learn": enable_learning})
        waiting = true
    while ws.get_available_packet_count() > 0:
        var raw := ws.get_packet().get_string_from_utf8()
        var message = JSON.parse_string(raw)
        if typeof(message) != TYPE_DICTIONARY:
            continue
        match message.get("type", ""):
            "created":
                registered = true
                waiting = false
            "action":
                if message.get("agentId") != agent_id:
                    continue
                _move(str(message.get("name", "idle")))
                var distance: float = global_position.distance_to(target.global_position)
                last_reward = clampf(previous_distance - distance, -1.0, 1.0)
                previous_distance = distance
                waiting = false
            "error":
                push_warning("Engine Bridge error: " + str(message.get("code", "unknown")))
                waiting = false
    next_step -= delta
    if registered and not waiting and next_step <= 0.0:
        next_step = step_seconds
        waiting = true
        _send({"type": "step", "agentId": agent_id,
            "observation": _sense(), "reward": last_reward})

func _sense() -> Array:
    var pos := global_position
    var offset := target.global_position - pos
    return [clampf(pos.x / 50.0, -1.0, 1.0), clampf(pos.z / 50.0, -1.0, 1.0),
        clampf(offset.x / 50.0, -1.0, 1.0), clampf(offset.z / 50.0, -1.0, 1.0),
        clampf(offset.length() / 70.0, 0.0, 1.0), 1.0]

func _move(action: String) -> void:
    var direction := Vector3.ZERO
    match action:
        "north": direction = Vector3.FORWARD
        "south": direction = Vector3.BACK
        "east": direction = Vector3.RIGHT
        "west": direction = Vector3.LEFT
    global_position += direction * (speed * step_seconds)

func _send(data: Dictionary) -> void:
    ws.send_text(JSON.stringify(data))

func _exit_tree() -> void:
    if ws.get_ready_state() == WebSocketPeer.STATE_OPEN:
        ws.close()
