import carla
import numpy as np
import time
import threading
import signal
import math
import cv2
from flask import Flask, Response, request, jsonify, render_template_string

app = Flask(__name__)

CONFIG = {
    "carla_port": 2000,
    "web_port": 5000,
    "camera_resolution": {"width": 1920, "height": 1080}
}

controller_data = {
    'throttle': 0.0,
    'steer': 0.0,
    'brake': 0.0,
    'reverse': False,
    'handbrake': False,
    'reset': False,
    'quit': False,
    'frame': None,
    'frame_lock': threading.Lock(),
    'gnss_data': (0.0, 0.0, 0.0),
    'imu_data': (0.0, 0.0, 0.0)
}

class HeroVehicle:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.role_name = "hero"
        self.original_spawn_point = None
        self._manual_control_active = False

    def follow_ego_with_spectator(self):
        transform = self.vehicle.get_transform()
        forward = transform.get_forward_vector()
        spectator_transform = carla.Transform(
            carla.Location(
                x=transform.location.x - 8.0 * forward.x,
                y=transform.location.y - 8.0 * forward.y,
                z=transform.location.z + 4.0,
            ),
            carla.Rotation(pitch=-15.0, yaw=transform.rotation.yaw, roll=0.0),
        )
        self.world.get_spectator().set_transform(spectator_transform)

    def initialize(self):

        try:
            print("Connecting to CARLA server...")
            self.client = carla.Client('localhost', CONFIG["carla_port"])
            self.client.set_timeout(20.0)
            self.world = self.client.get_world()

            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.world.apply_settings(settings)

            print("CARLA server connected")
            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}")
            return False

    def spawn_hero_vehicle(self):

        print("Spawning hero vehicle...")
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.lincoln.mkz_2017')
        if not vehicle_bp:
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))

        vehicle_bp.set_attribute('role_name', self.role_name)

        spawn_point = carla.Transform(
          carla.Location(x=-50, y=-136.00, z=0.5),
          carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

        self.original_spawn_point = spawn_point

        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if not self.vehicle:
            print("Error: Failed to spawn hero vehicle")
            return False

        print("Hero vehicle spawned successfully")
        return True

    def reset_vehicle_position(self):

        if self.vehicle and self.original_spawn_point:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.steer = 0.0
            control.brake = 1.0
            control.hand_brake = True
            self.vehicle.apply_control(control)

            self.vehicle.set_transform(self.original_spawn_point)
            print("\nVehicle reset to initial position")
            return True
        return False

    def setup_sensors(self):

        print("Initializing sensors...")

        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(CONFIG["camera_resolution"]["width"]))
        camera_bp.set_attribute('image_size_y', str(CONFIG["camera_resolution"]["height"]))
        camera_bp.set_attribute('fov', '90')

        camera_transform = carla.Transform(
            carla.Location(x=-5.0, z=3.0),
            carla.Rotation(pitch=-15.0)
        )
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        gnss_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        gnss_transform = carla.Transform(carla.Location(x=0.5, z=1.8))
        self.gnss_sensor = self.world.spawn_actor(gnss_bp, gnss_transform, attach_to=self.vehicle)

        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        imu_transform = carla.Transform(carla.Location(x=0.5, z=1.8))
        self.imu_sensor = self.world.spawn_actor(imu_bp, imu_transform, attach_to=self.vehicle)

        def camera_callback(image):

            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            with controller_data['frame_lock']:
                controller_data['frame'] = array.copy()

        def gnss_callback(data):

            controller_data['gnss_data'] = (data.latitude, data.longitude, data.altitude)

        def imu_callback(data):

            accel = (data.accelerometer.x, data.accelerometer.y, data.accelerometer.z)
            gyro = (data.gyroscope.x, data.gyroscope.y, data.gyroscope.z)
            compass = math.degrees(data.compass)
            controller_data['imu_data'] = (accel, gyro, compass)

        self.camera.listen(camera_callback)
        self.gnss_sensor.listen(gnss_callback)
        self.imu_sensor.listen(imu_callback)

        print("Sensors initialized")
        return True

    def cleanup(self):

        print("\nCleaning up resources...")
        destroy_list = [self.camera, self.gnss_sensor, self.imu_sensor, self.vehicle]
        for actor in destroy_list:
            if actor and actor.is_alive:
                actor.destroy()

        if self.world:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

        print("Resources cleaned up")

def generate_frames():

    while True:
        with controller_data['frame_lock']:
            if controller_data['frame'] is not None:
                frame = controller_data['frame']
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control', methods=['POST'])
def control():

    data = request.json
    controller_data['throttle'] = float(data.get('throttle', 0))
    controller_data['steer'] = float(data.get('steer', 0))
    controller_data['brake'] = float(data.get('brake', 0))
    controller_data['reverse'] = bool(data.get('reverse', False))
    controller_data['handbrake'] = bool(data.get('handbrake', False))
    controller_data['reset'] = bool(data.get('reset', False))
    controller_data['quit'] = bool(data.get('quit', False))
    return jsonify({'status': 'success'})

@app.route('/data')
def get_data():

    return jsonify({
        'gnss': controller_data['gnss_data'],
        'imu': controller_data['imu_data']
    })

@app.route('/')
def index():

    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>CARLA 控制中心</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.0/font/bootstrap-icons.css">
        <style>
            body { font-family: 'Segoe UI', sans-serif; margin: 0; background-color:

            .btn-carla { width: 100%; margin-bottom: 15px; padding: 12px; font-weight: 600; border: none; transition: all 0.3s; position: relative; }
            .btn-primary { background-color:
            .btn-primary:hover { background-color:
            .btn-danger { background-color:
            .btn-danger:hover { background-color:
            .btn-success { background-color:
            .btn-success:hover { background-color:
            .panel-title { color:
            .status-indicator { width: 15px; height: 15px; border-radius: 50%; display: inline-block; margin-right: 10px; }
            .status-online { background-color:
            .status-offline { background-color:
            .data-display { background-color:
            .control-group { margin-bottom: 20px; }
            .control-title { font-weight: bold; margin-bottom: 10px; }

        </style>
    </head>
    <body>
        <div id="app-container">
            <div id="control-panel">
                <h4 class="panel-title">车辆控制</h4>

                <div class="control-group">
                    <div class="control-title">驾驶控制</div>
                    <button class="btn btn-carla btn-primary" id="throttle-btn">加速 (W/↑)</button>
                    <button class="btn btn-carla btn-danger" id="brake-btn">刹车 (S/↓)</button>
                    <button class="btn btn-carla btn-success" id="left-btn">左转 (A/←)</button>
                    <button class="btn btn-carla btn-success" id="right-btn">右转 (D/→)</button>
                    <button class="btn btn-carla btn-warning" id="handbrake-btn">手刹 (空格)</button>
                </div>

                <div class="control-group">
                    <div class="control-title">操作</div>
                    <button class="btn btn-carla btn-info" id="reset-btn">重置位置 (R)</button>
                    <button class="btn btn-carla btn-danger" id="quit-btn">退出 (Q)</button>
                </div>

                <div class="control-group">
                    <div class="control-title">车辆数据</div>
                    <div class="data-display" id="data-display">
                        等待数据...
                    </div>
                </div>
            </div>

            <div id="view-container">
                <img id="carlaView" src="{{ url_for('video_feed') }}">
                <button id="fullscreen-btn">全屏</button>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // 控制状态
            const controls = {
                throttle: 0,
                steer: 0,
                brake: 0,
                reverse: false,
                handbrake: false,
                reset: false,
                quit: false
            };

            // 按钮元素
            const throttleBtn = document.getElementById('throttle-btn');
            const brakeBtn = document.getElementById('brake-btn');
            const leftBtn = document.getElementById('left-btn');
            const rightBtn = document.getElementById('right-btn');
            const handbrakeBtn = document.getElementById('handbrake-btn');
            const resetBtn = document.getElementById('reset-btn');
            const quitBtn = document.getElementById('quit-btn');
            const fullscreenBtn = document.getElementById('fullscreen-btn');
            const dataDisplay = document.getElementById('data-display');

            // 键盘状态
            const keys = {
                'w': false, 'ArrowUp': false,
                's': false, 'ArrowDown': false,
                'a': false, 'ArrowLeft': false,
                'd': false, 'ArrowRight': false,
                ' ': false, 'r': false, 'q': false
            };

            // 全屏功能
            function toggleFullscreen() {
                const elem = document.getElementById('view-container');
                if (!document.fullscreenElement) {
                    elem.requestFullscreen().catch(err => {
                        console.error(`全屏错误: ${err.message}`);
                    });
                } else {
                    document.exitFullscreen();
                }
            }

            // 更新控制命令
            function updateControls() {
                fetch('/control', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(controls)
                });
            }

            // 更新数据展示
            function updateData() {
                fetch('/data')
                    .then(response => response.json())
                    .then(data => {
                        const gnss = data.gnss;
                        const imu = data.imu;
                        dataDisplay.innerHTML = `
                            <strong>GNSS数据:</strong><br>
                            纬度: ${gnss[0].toFixed(6)}<br>
                            经度: ${gnss[1].toFixed(6)}<br>
                            海拔: ${gnss[2].toFixed(2)}m<br><br>
                            <strong>IMU数据:</strong><br>
                            加速度: (${imu[0][0].toFixed(2)}, ${imu[0][1].toFixed(2)}, ${imu[0][2].toFixed(2)})<br>
                            陀螺仪: (${imu[1][0].toFixed(2)}, ${imu[1][1].toFixed(2)}, ${imu[1][2].toFixed(2)})<br>
                            罗盘: ${imu[2].toFixed(2)}°
                        `;
                    });
            }

            // 按钮事件监听
            throttleBtn.addEventListener('mousedown', () => { controls.throttle = 1; updateControls(); });
            throttleBtn.addEventListener('mouseup', () => { controls.throttle = 0; updateControls(); });
            throttleBtn.addEventListener('mouseleave', () => { controls.throttle = 0; updateControls(); });

            brakeBtn.addEventListener('mousedown', () => {
                controls.brake = 1;
                controls.reverse = false;
                updateControls();
            });
            brakeBtn.addEventListener('mouseup', () => {
                controls.brake = 0;
                updateControls();
            });
            brakeBtn.addEventListener('mouseleave', () => {
                controls.brake = 0;
                updateControls();
            });

            // 长按刹车进入倒车模式
            let brakeTimer;
            brakeBtn.addEventListener('mousedown', () => {
                brakeTimer = setTimeout(() => {
                    controls.reverse = true;
                    controls.brake = 0;
                    updateControls();
                }, 500);
            });
            brakeBtn.addEventListener('mouseup', () => {
                clearTimeout(brakeTimer);
            });

            leftBtn.addEventListener('mousedown', () => { controls.steer = -0.5; updateControls(); });
            leftBtn.addEventListener('mouseup', () => { controls.steer = 0; updateControls(); });
            leftBtn.addEventListener('mouseleave', () => { controls.steer = 0; updateControls(); });

            rightBtn.addEventListener('mousedown', () => { controls.steer = 0.5; updateControls(); });
            rightBtn.addEventListener('mouseup', () => { controls.steer = 0; updateControls(); });
            rightBtn.addEventListener('mouseleave', () => { controls.steer = 0; updateControls(); });

            handbrakeBtn.addEventListener('mousedown', () => { controls.handbrake = true; updateControls(); });
            handbrakeBtn.addEventListener('mouseup', () => { controls.handbrake = false; updateControls(); });
            handbrakeBtn.addEventListener('mouseleave', () => { controls.handbrake = false; updateControls(); });

            resetBtn.addEventListener('click', () => {
                controls.reset = true;
                updateControls();
                setTimeout(() => { controls.reset = false; updateControls(); }, 100);
            });

            quitBtn.addEventListener('click', () => {
                controls.quit = true;
                updateControls();
                setTimeout(() => { controls.quit = false; updateControls(); }, 100);
            });

            fullscreenBtn.addEventListener('click', toggleFullscreen);

            // 键盘事件监听
            document.addEventListener('keydown', (e) => {
                if (keys.hasOwnProperty(e.key)) {
                    keys[e.key] = true;

                    if (e.key === 'w' || e.key === 'ArrowUp') {
                        controls.throttle = 1;
                    }
                    if (e.key === 's' || e.key === 'ArrowDown') {
                        controls.brake = 1;
                        controls.reverse = false;
                    }
                    if (e.key === 'a' || e.key === 'ArrowLeft') {
                        controls.steer = -0.5;
                    }
                    if (e.key === 'd' || e.key === 'ArrowRight') {
                        controls.steer = 0.5;
                    }
                    if (e.key === ' ') {
                        controls.handbrake = true;
                    }
                    if (e.key === 'r') {
                        controls.reset = true;
                        setTimeout(() => { controls.reset = false; }, 100);
                    }
                    if (e.key === 'q') {
                        controls.quit = true;
                        setTimeout(() => { controls.quit = false; }, 100);
                    }

                    updateControls();
                }
            });

            document.addEventListener('keyup', (e) => {
                if (keys.hasOwnProperty(e.key)) {
                    keys[e.key] = false;

                    if (e.key === 'w' || e.key === 'ArrowUp') {
                        controls.throttle = 0;
                    }
                    if (e.key === 's' || e.key === 'ArrowDown') {
                        controls.brake = 0;
                        controls.reverse = false;
                    }
                    if (e.key === 'a' || e.key === 'ArrowLeft') {
                        controls.steer = 0;
                    }
                    if (e.key === 'd' || e.key === 'ArrowRight') {
                        controls.steer = 0;
                    }
                    if (e.key === ' ') {
                        controls.handbrake = false;
                    }

                    updateControls();
                }
            });

            // 长按刹车进入倒车模式(键盘)
            let keyBrakeTimer;
            document.addEventListener('keydown', (e) => {
                if (e.key === 's' || e.key === 'ArrowDown') {
                    keyBrakeTimer = setTimeout(() => {
                        controls.reverse = true;
                        controls.brake = 0;
                        updateControls();
                    }, 500);
                }
            });

            document.addEventListener('keyup', (e) => {
                if (e.key === 's' || e.key === 'ArrowDown') {
                    clearTimeout(keyBrakeTimer);
                }
            });

            // 定期更新数据
            setInterval(updateData, 500);

            // 初始数据加载
            updateData();
        </script>
    </body>
    </html>
