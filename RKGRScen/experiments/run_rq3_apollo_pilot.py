RQ3 Apollo+CARLA pilot runner.

运行位置：宿主机 carla-clean 虚拟环境。
默认只执行 RQ2 Full 中已成功触发的 Town03 / scenario_00126，用于验证：
- CARLA 场景部署
- Apollo routing 请求
- Apollo 自主控制 ego
- CARLA 侧 NPC/障碍物控制
- 轨迹采样与统一检测器复用

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import carla

ROOT = Path("/home/zxy/apollo/data/test/point2")
RKGRSCEN_ROOT = ROOT / "RKGRScen"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from RKGRScen.execution.violation_detector import detect_violation
from RKGRScen.experiments.run_rq2_ablation import retrieval_to_solver_result
from RKGRScen.query.constraint_solver import ConstraintSolver

DEFAULT_CASE = RKGRSCEN_ROOT / "data/evaluation/rq2_balanced_420_execution/full/case_results/scenario_00126.json"
DEFAULT_OUT = RKGRSCEN_ROOT / "data/evaluation/rq3_apollo_pilot/scenario_00126_pilot.json"
APOLLO_DOCKER = "apollo_dev_zxy"
RESET_HELPER = Path("/home/zxy/apollo/data/test/test/reset.py")

def reset_ego_vehicle(vehicle: carla.Vehicle, world: carla.World) -> bool:
    if not RESET_HELPER.exists():
        return False
    spec = importlib.util.spec_from_file_location("rq3_reset_helper", str(RESET_HELPER))
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reset_fn = getattr(module, "reset_vehicle_to_fixed_position", None)
    if reset_fn is None:
        return False
    return bool(reset_fn(vehicle, world, reset_transform=None))

def _debug_report(hypothesis_id: str, msg: str, data: Dict[str, Any]) -> None:
    server_url = os.environ.get("DEBUG_SERVER_URL") or "http://127.0.0.1:7777/event"
    session_id = os.environ.get("DEBUG_SESSION_ID") or "apollo-ego-static"
    payload = {
        "sessionId": session_id,
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": "run_rq3_apollo_pilot.py",
        "msg": msg,
        "data": data,
        "ts": int(time.time() * 1000),
    }
    try:
        import urllib.request
        req = urllib.request.Request(server_url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2).read()
    except Exception:
        pass

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)

def normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle

def speed_mps(actor: carla.Actor) -> float:
    v = actor.get_velocity()
    return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)

def dist2d(a: carla.Location, b: carla.Location) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def carla_to_apollo_pose(transform: carla.Transform, shift: float = 1.355) -> Dict[str, float]:
    heading = -math.radians(transform.rotation.yaw)
    return {
        "x": float(transform.location.x - shift * math.cos(heading)),
        "y": float(-transform.location.y - shift * math.sin(heading)),
        "z": float(transform.location.z),
        "heading": float(heading),
    }

def apollo_point_from_carla_location(location: carla.Location, heading: float = 0.0) -> Dict[str, float]:
    return {
        "x": float(location.x),
        "y": float(-location.y),
        "z": float(location.z),
        "heading": float(heading),
    }

def run_apollo_pose_probe(timeout_s: float = 8.0) -> Optional[Dict[str, float]]:
    code = r'''
from cyber.python.cyber_py3 import cyber
from modules.common_msgs.localization_msgs.localization_pb2 import LocalizationEstimate
import time
cyber.init()
node = cyber.Node("rq3_pose_once")
box = {"pose": None}
def cb(msg):
    p = msg.pose.position
    print("POSE_JSON={\"x\":%.9f,\"y\":%.9f,\"z\":%.9f,\"heading\":%.12f}" % (p.x, p.y, p.z, msg.pose.heading), flush=True)
    box["pose"] = True
node.create_reader("/apollo/localization/pose", LocalizationEstimate, cb)
start = time.time()
while time.time() - start < 5.0 and box["pose"] is None:
    time.sleep(0.05)
cyber.shutdown()

    cmd = [
        "docker", "exec", "-u", "zxy", APOLLO_DOCKER,
        "bash", "-ic",
        "export PYTHONPATH=/apollo:$PYTHONPATH; export PYTHONPATH=/apollo/cyber/python:$PYTHONPATH; export PYTHONPATH=/apollo/modules/tools:$PYTHONPATH; export PYTHONPATH=/apollo/bazel-bin:$PYTHONPATH; export PYTHONIOENCODING=utf-8; "
        f"python3 -c {json.dumps(code)}",
    ]
    proc = subprocess.run(cmd, cwd="/home/zxy/apollo", text=True, capture_output=True, timeout=timeout_s)
    for line in proc.stdout.splitlines():
        if line.startswith("POSE_JSON="):
            return json.loads(line.split("=", 1)[1])
    return None

def _run_dreamview_ws(payload: Dict[str, Any], websocket_url: str, recv_timeout_s: float = 3.0) -> Dict[str, Any]:
    code = (
        "import asyncio\n"
        "import json\n"
        "import websockets\n\n"
        f"async def main():\n"
        f"    async with websockets.connect({json.dumps(websocket_url)}) as websocket:\n"
        f"        await websocket.send({json.dumps(json.dumps(payload))})\n"
        "        try:\n"
        f"            response = await asyncio.wait_for(websocket.recv(), timeout={float(recv_timeout_s)})\n"
        "            print(response)\n"
        "        except Exception:\n"
        "            print('')\n\n"
        "asyncio.get_event_loop().run_until_complete(main())\n"
    )
    cmd = [
        "docker", "exec", "-i", "-u", "zxy", APOLLO_DOCKER,
        "bash", "-ic",
        "export PYTHONPATH=/apollo:$PYTHONPATH; export PYTHONPATH=/apollo/cyber/python:$PYTHONPATH; export PYTHONPATH=/apollo/modules/tools:$PYTHONPATH; export PYTHONPATH=/apollo/bazel-bin:$PYTHONPATH; export PYTHONIOENCODING=utf-8; python3 -",
    ]
    proc = subprocess.run(cmd, cwd="/home/zxy/apollo", text=True, input=code, capture_output=True, timeout=20)
    return {
        "request": payload,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

def send_hmi_action(action: str, value: Optional[str] = None, websocket_url: str = "ws://127.0.0.1:8888/websocket") -> Dict[str, Any]:
    payload = {"type": "HMIAction", "action": action}
    if value is not None:
        payload["value"] = value
    return _run_dreamview_ws(payload, websocket_url)

def send_routing(start: Dict[str, float], end: Dict[str, float], websocket_url: str) -> Dict[str, Any]:
    payload = {
        "type": "SendRoutingRequest",
        "start": {
            "x": float(start["x"]),
            "y": float(start["y"]),
            "z": float(start["z"]),
            "heading": float(start["heading"]),
        },
        "end": {
            "x": float(end["x"]),
            "y": float(end["y"]),
            "z": float(end["z"]),
            "heading": float(end.get("heading", 0.0)),
        },
        "waypoint": [],
    }
    code = (
        "import asyncio\n"
        "import json\n"
        "import websockets\n\n"
        f"async def main():\n"
        f"    async with websockets.connect({json.dumps(websocket_url)}) as websocket:\n"
        f"        await websocket.send({json.dumps(json.dumps(payload))})\n"
        "        response = await websocket.recv()\n"
        "        print(response)\n\n"
        "asyncio.get_event_loop().run_until_complete(main())\n"
    )
    cmd = [
        "docker", "exec", "-i", "-u", "zxy", APOLLO_DOCKER,
        "bash", "-ic",
        "export PYTHONPATH=/apollo:$PYTHONPATH; export PYTHONPATH=/apollo/cyber/python:$PYTHONPATH; export PYTHONPATH=/apollo/modules/tools:$PYTHONPATH; export PYTHONPATH=/apollo/bazel-bin:$PYTHONPATH; export PYTHONIOENCODING=utf-8; python3 -",
    ]
    proc = subprocess.run(cmd, cwd="/home/zxy/apollo", text=True, input=code, capture_output=True, timeout=20)
    return {
        "request": payload,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }

def scenario_from_manifest_row(row: Dict[str, Any]) -> Dict[str, Any]:
    source_path = Path(str(row.get("source_path") or row.get("original_source_path")))
    source = load_json(source_path)
    spec = row.get("retrieval", {}).get("scenario_spec") or source.get("dsl") or {}
    retrieval = row.get("retrieval") or {}
    if not spec:
        raise RuntimeError(f"manifest 行缺少 scenario_spec: {row.get('scenario_id')}")
    if not retrieval.get("local_top_k"):
        raise RuntimeError(f"manifest 行缺少 retrieval.local_top_k: {row.get('scenario_id')}")
    solver_retrieval = retrieval_to_solver_result(retrieval, str(row.get("source_violation_type") or row.get("violation_type") or source.get("violation_type", "")))
    config = ConstraintSolver().solve(spec, [solver_retrieval])
    config.scenario_id = str(row["scenario_id"])
    scenario = config.to_dict()
    scenario["source_path"] = str(source_path)
    scenario["manifest_row"] = row
    return scenario

def set_weather(world: carla.World, environment: Dict[str, Any]) -> None:
    weather = str(environment.get("weather", "clear")).lower()
    if weather in {"clear", "day", "sunny"}:
        world.set_weather(carla.WeatherParameters.ClearNoon)
    elif "rain" in weather:
        world.set_weather(carla.WeatherParameters.WetNoon)
    elif "cloud" in weather:
        world.set_weather(carla.WeatherParameters.CloudyNoon)

def waypoint_transform(world: carla.World, spec: Dict[str, Any], z_offset: float = 0.5) -> Tuple[carla.Waypoint, carla.Transform]:
    wp = world.get_map().get_waypoint_xodr(int(spec["road_id"]), int(spec["lane_id"]), float(spec["s"]))
    if wp is None:
        raise RuntimeError(f"无法解析 waypoint: {spec}")
    tf = carla.Transform(
        carla.Location(wp.transform.location.x, wp.transform.location.y, wp.transform.location.z + z_offset),
        wp.transform.rotation,
    )
    return wp, tf

def ensure_ego_vehicle(world: carla.World, ego_tf: carla.Transform) -> carla.Vehicle:
    vehicles = list(world.get_actors().filter("vehicle.*"))
    for vehicle in vehicles:
        if vehicle.attributes.get("role_name") == "hero":
            return vehicle
    blueprints = world.get_blueprint_library()
    candidates = blueprints.filter("vehicle.lincoln.mkz*") or blueprints.filter("vehicle.lincoln.*") or blueprints.filter("vehicle.*")
    if not candidates:
        raise RuntimeError("CARLA 中没有可用于 ego 的车辆蓝图")
    bp = candidates[0]
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "hero")
    ego = world.try_spawn_actor(bp, ego_tf)
    if ego is None:
        raise RuntimeError(f"ego 车辆生成失败: {bp.id}")
    return ego

def find_hero(world: carla.World) -> carla.Vehicle:
    vehicles = list(world.get_actors().filter("vehicle.*"))
    for vehicle in vehicles:
        if vehicle.attributes.get("role_name") == "hero":
            return vehicle
    if vehicles:
        return vehicles[0]
    raise RuntimeError("CARLA 中没有 ego/hero 车辆")

def spawn_npcs(world: carla.World, scenario: Dict[str, Any]) -> List[Tuple[Dict[str, Any], carla.Vehicle]]:
    blueprints = world.get_blueprint_library()
    actors = []
    for npc in scenario.get("npcs", []):
        _, tf = waypoint_transform(world, npc["spawn_waypoint"])
        model = "vehicle.audi.a2"
        bp = blueprints.filter(model)[0] if blueprints.filter(model) else blueprints.filter("vehicle.*")[0]
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", f"rq3_{npc.get('id', 'npc')}")
        actor = world.try_spawn_actor(bp, tf)
        if actor is None:
            raise RuntimeError(f"NPC 生成失败: {npc}")
        if "Static Obstacle" in str(npc.get("behavior", "")):
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        actors.append((npc, actor))
    return actors

def sample_tick(
    timestamp_s: float,
    world_map: carla.Map,
    ego: carla.Vehicle,
    npcs: List[Tuple[Dict[str, Any], carla.Vehicle]],
    scenario: Dict[str, Any],
) -> Dict[str, Any]:
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_speed = speed_mps(ego)
    conflict = scenario["conflict_point"]
    conflict_loc = carla.Location(float(conflict["x"]), float(conflict["y"]), ego_loc.z)
    npc_rows = []
    for npc_cfg, npc_actor in npcs:
        npc_tf = npc_actor.get_transform()
        npc_loc = npc_tf.location
        npc_speed = speed_mps(npc_actor)
        gap = dist2d(ego_loc, npc_loc)
        closing = max(ego_speed - npc_speed, 0.0)
        ttc = gap / max(closing, 0.1)
        npc_wp = world_map.get_waypoint(npc_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        relative_x = npc_loc.x - ego_loc.x
        relative_y = npc_loc.y - ego_loc.y
        yaw_rad = math.radians(ego_tf.rotation.yaw)
        relative_longitudinal = relative_x * math.cos(yaw_rad) + relative_y * math.sin(yaw_rad)
        npc_rows.append({
            "id": npc_cfg.get("id"),
            "role": npc_cfg.get("role"),
            "speed_mps": round(npc_speed, 3),
            "distance_to_conflict_m": round(dist2d(npc_loc, conflict_loc), 3),
            "distance_to_ego_m": round(gap, 3),
            "ttc_to_ego_s": round(ttc, 3),
            "location": {"x": round(npc_loc.x, 3), "y": round(npc_loc.y, 3)},
            "yaw": round(npc_tf.rotation.yaw, 3),
            "heading_gap_to_ego_deg": round(abs(normalize_angle_deg(npc_tf.rotation.yaw - ego_tf.rotation.yaw)), 3),
            "road_id": getattr(npc_wp, "road_id", None),
            "lane_id": getattr(npc_wp, "lane_id", None),
            "source_lane_id": npc_cfg.get("spawn_waypoint", {}).get("lane_id"),
            "target_lane_id": npc_cfg.get("target_lane_id"),
            "lane_change_phase": "stopped" if "Static Obstacle" in str(npc_cfg.get("behavior", "")) else "lane_follow",
            "lane_change_completed": False,
            "relative_longitudinal_m": round(relative_longitudinal, 3),
        })
    return {
        "timestamp_s": round(timestamp_s, 3),
        "ego": {
            "role": scenario.get("ego", {}).get("role", "ego"),
            "speed_mps": round(ego_speed, 3),
            "distance_to_conflict_m": round(dist2d(ego_loc, conflict_loc), 3),
            "distance_to_stop_line_m": round(dist2d(ego_loc, conflict_loc), 3),
            "passed_stop_line": False,
            "signal_state": "green",
            "location": {"x": round(ego_loc.x, 3), "y": round(ego_loc.y, 3)},
            "yaw": round(ego_tf.rotation.yaw, 3),
            "apollo_pose_estimate": carla_to_apollo_pose(ego_tf),
        },
        "npcs": npc_rows,
        "conflict_point": conflict,
        "traffic_state": {"signal": "green", "expected_violation": scenario["violation_type"]},
    }

def drive_static_npcs(npcs: List[Tuple[Dict[str, Any], carla.Vehicle]]) -> None:
    for cfg, actor in npcs:
        if "Static Obstacle" in str(cfg.get("behavior", "")):
            actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=False))

def run_once(args: argparse.Namespace) -> Dict[str, Any]:
    if args.manifest_row_json:
        scenario = scenario_from_manifest_row(json.loads(args.manifest_row_json))
        input_path = scenario.get("source_path", "manifest_row")
    else:
        case_payload = load_json(Path(args.case))
        scenario = case_payload["scenario_config"]
        input_path = str(args.case)
    _debug_report("A", "[DEBUG] loaded scenario", {"input": input_path, "scenario_id": scenario.get("scenario_id"), "violation_type": scenario.get("violation_type"), "map_name": scenario.get("map_name"), "ego_spawn_waypoint": scenario.get("ego", {}).get("spawn_waypoint"), "route_distance_m": args.route_distance_m})

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    target_map = scenario["map_name"].split("/")[-1]
    current_map = world.get_map().name.split("/")[-1]
    if current_map != target_map:
        if args.allow_load_world:
            world = client.load_world(target_map)
            time.sleep(3.0)
        else:
            raise RuntimeError(f"当前 CARLA 地图为 {current_map}，pilot 需要 {target_map}；未启用 --allow-load-world")

    set_weather(world, scenario.get("environment", {}))
    world_map = world.get_map()
    ego_wp, ego_tf = waypoint_transform(world, scenario["ego"]["spawn_waypoint"])
    ego = ensure_ego_vehicle(world, ego_tf)
    hero = ego
    _debug_report("B", "[DEBUG] carla world and hero resolved", {"current_map": world_map.name, "hero_id": getattr(hero, 'id', None), "hero_role": hero.attributes.get('role_name', None), "hero_transform": str(hero.get_transform()), "ego_waypoint": scenario.get("ego", {}).get("spawn_waypoint")})
    route_candidates = ego_wp.next(float(args.route_distance_m))
    route_wp = route_candidates[0] if route_candidates else ego_wp
    route_tf = route_wp.transform

    spawned: List[Tuple[Dict[str, Any], carla.Vehicle]] = []
    created_ego = hero.id
    ticks: List[Dict[str, Any]] = []
    routing_info: Dict[str, Any] = {}
    try:
        hero.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
        hero.set_transform(ego_tf)
        hero.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        _debug_report("C", "[DEBUG] ego reset and locked", {"ego_transform": str(ego_tf), "hero_control": {"throttle": 0.0, "brake": 1.0, "hand_brake": True}})
        time.sleep(args.settle_s)

        hmi_vehicle = send_hmi_action("CHANGE_VEHICLE", "Lincoln2017MKZ LGSVL", websocket_url=args.websocket_url)
        hmi_setup = send_hmi_action("SETUP_MODE", websocket_url=args.websocket_url)
        hmi_auto = send_hmi_action("ENTER_AUTO_MODE", websocket_url=args.websocket_url)
        _debug_report("D", "[DEBUG] dreamview hmi actions issued", {"vehicle": hmi_vehicle, "setup": hmi_setup, "auto": hmi_auto})
        time.sleep(1.0)

        hero.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
        _debug_report("E", "[DEBUG] ego control released", {"ego_transform": str(ego_tf), "hero_control": {"throttle": 0.0, "brake": 0.0, "hand_brake": False}})
        apollo_start = run_apollo_pose_probe()
        if apollo_start is None:
            apollo_start = carla_to_apollo_pose(hero.get_transform())
        _debug_report("F", "[DEBUG] apollo start pose sampled", {"apollo_start": apollo_start})

        apollo_end = apollo_point_from_carla_location(route_tf.location, heading=-math.radians(route_tf.rotation.yaw))
        _debug_report("G", "[DEBUG] apollo route end computed", {"route_tf": str(route_tf), "apollo_end": apollo_end})
        spawned = spawn_npcs(world, scenario)
        time.sleep(0.5)

        routing_info = send_routing(apollo_start, apollo_end, args.websocket_url)
        if routing_info.get("returncode") != 0:
            raise RuntimeError(f"Apollo routing 发送失败: {routing_info}")
        _debug_report("H", "[DEBUG] routing finished", {"routing_info": routing_info})

        start_time = time.time()
        while time.time() - start_time <= args.duration_s:
            drive_static_npcs(spawned)
            world.wait_for_tick(seconds=1.0)
            ts = time.time() - start_time
            ticks.append(sample_tick(ts, world_map, hero, spawned, scenario))
            time.sleep(args.step_s)
    finally:
        for _, actor in spawned:
            try:
                actor.destroy()
            except RuntimeError:
                pass
        try:
            reset_ego_vehicle(hero, world)
        except Exception:
            pass

    violation = detect_violation(
        scenario["violation_type"],
        ticks,
        scenario.get("expected_violation", {}).get("params", {}),
    )
    max_ego_speed_mps = 0.0
    ego_displacement_m = 0.0
    if ticks:
        max_ego_speed_mps = max(float(t.get("ego", {}).get("speed_mps", 0.0)) for t in ticks)
        first_loc = ticks[0].get("ego", {}).get("location", {})
        last_loc = ticks[-1].get("ego", {}).get("location", {})
        ego_displacement_m = math.hypot(
            float(last_loc.get("x", 0.0)) - float(first_loc.get("x", 0.0)),
            float(last_loc.get("y", 0.0)) - float(first_loc.get("y", 0.0)),
        )
    execution_success = len(ticks) > 0 and max_ego_speed_mps >= 0.5 and ego_displacement_m >= 2.0
    result = {
        "rq": "RQ3",
        "runner": "apollo_carla_pilot",
        "input_path": input_path,
        "scenario_config": scenario,
        "execution_status": {
            "success": execution_success,
            "tick_count": len(ticks),
            "max_ego_speed_mps": round(max_ego_speed_mps, 3),
            "ego_displacement_m": round(ego_displacement_m, 3),
            "failure_reason": "" if execution_success else "ego did not move sufficiently under Apollo control",
        },
        "apollo_routing": routing_info,
        "execution_trace": {
            "scenario_id": scenario["scenario_id"],
            "ticks": ticks,
            "metadata": {
                "map": world.get_map().name,
                "tick_count": len(ticks),
                "ego_spawn_transform": str(ego_tf),
                "route_end_transform": str(route_tf),
                "coordinate_rule": "apollo_x=carla_x-1.355*cos(heading), apollo_y=-carla_y-1.355*sin(heading), apollo_heading=-carla_yaw_rad",
            },
        },
        "violation_result": violation,
    }
    return result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=str(DEFAULT_CASE))
    parser.add_argument("--manifest-row-json", default="")
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--websocket-url", default="ws://127.0.0.1:8888/websocket")
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--step-s", type=float, default=0.2)
    parser.add_argument("--settle-s", type=float, default=2.0)
    parser.add_argument("--route-distance-m", type=float, default=65.0)
    parser.add_argument("--allow-load-world", action="store_true")
    args = parser.parse_args()

    result = run_once(args)
    dump_json(Path(args.output), result)
    preview = {
        "scenario_id": result["scenario_config"]["scenario_id"],
        "violation_type": result["scenario_config"]["violation_type"],
        "tick_count": len(result["execution_trace"]["ticks"]),
        "execution_success": result["execution_status"]["success"],
        "max_ego_speed_mps": result["execution_status"]["max_ego_speed_mps"],
        "ego_displacement_m": result["execution_status"]["ego_displacement_m"],
        "detected": result["violation_result"].get("detected"),
        "reason": result["violation_result"].get("reason"),
        "output": str(args.output),
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
