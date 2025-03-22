import pigpio
import pygame as pg
import subprocess
import time
import threading
import os
from datetime import datetime
from picamera2 import Picamera2
import numpy as np
import cv2
import json

# 确保pigpiod在运行
def ensure_pigpiod_running():
    try:
        result = subprocess.run(['pgrep', 'pigpiod'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        if result.returncode != 0:
            print("pigpiod is not running. Starting it now...")
            subprocess.run(['sudo', 'pigpiod'], check=True)
        else:
            print("pigpiod is already running.")
    except Exception as e:
        print(f"Error ensuring pigpiod is running: {e}")

# 应用死区功能，避免控制器微小输入
def apply_deadzone(value, deadzone):
    """Applies a deadzone to joystick axis input."""
    if abs(value) < deadzone:
        return 0
    return value

# 检查控制器连接状态
def controller_connection_check():
    if pg.joystick.get_count() == 0:
        return False
    else:
        return True

# 数据收集类
class DataCollector:
    def __init__(self, frame_interval=5, base_dir="~/training_data"):
        # 基本设置
        self.frame_interval = frame_interval  # 每隔多少帧保存一次
        self.frame_count = 0
        self.collecting = False
        self.base_dir = os.path.expanduser(base_dir)
        self.session_dir = None
        self.steering_data = []
        
        # 确保基本目录存在
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
            print(f"Created base directory: {self.base_dir}")
        
        # 初始化硬件
        ensure_pigpiod_running()
        time.sleep(1)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            raise RuntimeError("Failed to connect to pigpio daemon.")
        
        # PWM设置
        self.pin_steering = 12  # GPIO Pin Number for steering
        self.pin_throttle = 13  # GPIO Pin Number for throttle
        self.pin_rec = 25       # GPIO Pin for recording indicator
        self.freq = 50          # PWM frequency in Hz
        
        # 设置PWM
        self.pi.set_PWM_range(self.pin_steering, 1000)
        self.pi.set_PWM_frequency(self.pin_steering, self.freq)
        self.pi.set_PWM_range(self.pin_throttle, 1000)
        self.pi.set_PWM_frequency(self.pin_throttle, self.freq)
        
        # 控制变量
        self.steering_ms_centered = 1.075  # 舵机中心点
        self.steering_ms_range = 0.1702   # 舵机运动范围
        self.steering_ms_out = self.steering_ms_centered
        self.steering_axis = 0.0
        
        self.throttle_ms_centered = 1.35  # 油门中心点
        self.throttle_ms_range = 0.06     # 油门运动范围
        self.throttle_ms_out = self.throttle_ms_centered
        self.throttle_axis = 0.0
        self.throttle_key = 0             # 数字传输管理
        
        # 初始化摄像头
        self.init_camera()
        
        # 初始化pygame
        pg.init()
        pg.joystick.init()
        self.clock = pg.time.Clock()
        self.deadzone = 0.1  # 控制器死区
        
        if controller_connection_check():
            self.joystick = pg.joystick.Joystick(0)
            self.joystick.init()
        else:
            self.joystick = None
            
        # 按钮状态跟踪
        self.button_states = {}
        self.input_value = 0  # 0 for keyboard, 1 for controller
        
        # 启动信号
        self.startup_signal()
    
    def startup_signal(self):
        """发出准备就绪的信号"""
        for _ in range(3):
            self.pi.write(self.pin_rec, 1)
            time.sleep(0.5)
            self.pi.write(self.pin_rec, 0)
            time.sleep(0.5)
    
    def init_camera(self):
        """初始化摄像头"""
        try:
            self.camera = Picamera2()
            # 配置相机设置，选择较低分辨率以提高性能
            self.camera_config = self.camera.create_video_configuration(
                main={"size": (320, 240), "format": 'RGB888'}
            )
            self.camera.configure(self.camera_config)
            self.camera.start()
            time.sleep(0.5)  # 给相机预热时间
            self.cam_connected = True
            print("Camera initialized successfully")
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            self.cam_connected = False
    
    def create_session_directory(self):
        """为当前会话创建一个新目录"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.session_dir = os.path.join(self.base_dir, f"session_{timestamp}")
        self.images_dir = os.path.join(self.session_dir, "images")
        
        os.makedirs(self.session_dir)
        os.makedirs(self.images_dir)
        print(f"Created session directory: {self.session_dir}")
    
    def start_collecting(self):
        """开始数据收集"""
        if self.collecting:
            print("Data collection already in progress")
            return
        
        if not self.cam_connected:
            print("Cannot start data collection: Camera not connected")
            return
        
        self.create_session_directory()
        self.collecting = True
        self.steering_data = []
        self.frame_count = 0
        self.pi.write(self.pin_rec, 1)  # 打开指示灯
        print("Started data collection")
    
    def stop_collecting(self):
        """停止数据收集"""
        if not self.collecting:
            print("No data collection in progress")
            return
        
        self.collecting = False
        self.pi.write(self.pin_rec, 0)  # 关闭指示灯
        
        # 保存转向数据
        if self.steering_data:
            data_file = os.path.join(self.session_dir, "steering_data.json")
            with open(data_file, 'w') as f:
                json.dump(self.steering_data, f)
            print(f"Saved {len(self.steering_data)} steering data points to {data_file}")
        
        print("Stopped data collection")
    
    def capture_frame(self):
        """捕获当前帧并保存图像和转向数据"""
        if not self.collecting or not self.cam_connected:
            return
        
        self.frame_count += 1
        
        # 每隔frame_interval帧保存一次
        if self.frame_count % self.frame_interval == 0:
            # 捕获图像
            frame = self.camera.capture_array()
            
            # 创建文件名
            frame_num = len(self.steering_data)
            image_path = os.path.join(self.images_dir, f"frame_{frame_num:05d}.jpg")
            
            # 保存图像
            cv2.imwrite(image_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # 保存当前的转向数据
            #steering_value = (self.steering_ms_out - self.steering_ms_centered) / self.steering_ms_range
            steering_value = self.steering_ms_out
            self.steering_data.append({
                "frame": frame_num,
                "steering_value": steering_value,
                "timestamp": datetime.now().isoformat()
            })
            
            print(f"Saved frame {frame_num} with steering value {steering_value:.3f}")
    
    def handle_one_shot_button(self, button_index):
        """检测按钮的一次性按下事件"""
        if self.joystick is None:
            return False
        
        # 初始化按钮状态（如果尚未跟踪）
        if button_index not in self.button_states:
            self.button_states[button_index] = False
        
        # 获取当前按钮状态
        button_current = self.joystick.get_button(button_index)
        
        # 检查一次性按下
        if button_current == 1 and not self.button_states[button_index]:
            self.button_states[button_index] = True
            return True
        elif button_current == 0 and self.button_states[button_index]:
            self.button_states[button_index] = False
        
        return False
    
    def controller_reconnection_handler(self):
        """处理控制器重新连接"""
        if pg.joystick.get_count() == 0:
            if self.joystick is not None:
                print("Controller disconnected. Please wait for reconnection.")
                self.joystick = None
        elif self.joystick is None:
            self.joystick = pg.joystick.Joystick(0)
            self.joystick.init()
            print("Controller reconnected")
    
    def get_keyboard_inputs(self):
        """获取键盘输入"""
        keys = pg.key.get_pressed()
        
        a = 0  # 转向
        b = 0  # 油门
        
        if keys[pg.K_LEFT]:
            a -= 1
        if keys[pg.K_RIGHT]:
            a += 1
        
        if keys[pg.K_DOWN]:
            b -= 1
        if keys[pg.K_UP]:
            b += 1
        
        if keys[pg.K_SPACE]:
            self.throttle_key = 1
        
        # 用R键开始/停止数据收集
        if keys[pg.K_r] and not self.r_key_pressed:
            self.r_key_pressed = True
            if self.collecting:
                self.stop_collecting()
            else:
                self.start_collecting()
        elif not keys[pg.K_r]:
            self.r_key_pressed = False
        
        self.steering_axis = a
        self.throttle_axis = b
    
    def get_controller_inputs(self):
        """获取控制器输入"""
        if self.joystick is None:
            return
        
        self.steering_axis = apply_deadzone(self.joystick.get_axis(0), self.deadzone)
        self.throttle_axis = self.joystick.get_axis(4)/2 + 0.5
        
        # 处理按钮输入
        if self.handle_one_shot_button(4):  # 假设按钮4用于开始/停止数据收集
            if self.collecting:
                self.stop_collecting()
            else:
                self.start_collecting()
        
        # 处理D-pad输入
        dpad = self.joystick.get_hat(0)
        if dpad[1] == 1:
            self.throttle_key = 1
        elif dpad[1] == -1:
            self.throttle_key = -0.625
        elif dpad[0] == 1 or dpad[0] == -1:
            self.throttle_key = 0
    
    def input_handler(self):
        """处理输入，优先处理键盘输入"""
        self.steering_axis = 0
        self.throttle_axis = 0
        
        keys = pg.key.get_pressed()
        if any(keys):
            self.input_value = 0
        else:
            self.input_value = 1
        
        if self.input_value == 0:
            self.get_keyboard_inputs()
        else:
            if controller_connection_check():
                self.get_controller_inputs()
    
    def steering_handler(self):
        """处理转向"""
        self.steering_ms_out = self.steering_ms_centered + self.steering_ms_range * self.steering_axis
    
    def throttle_handler(self):
        """处理油门"""
        self.throttle_ms_out = self.throttle_ms_centered + self.throttle_ms_range * self.throttle_axis * self.throttle_key
    
    def update_steering_duty(self):
        """更新转向PWM占空比"""
        duty = self.steering_ms_out / (1/self.pi.get_PWM_frequency(self.pin_steering) * 1000)
        self.pi.set_PWM_dutycycle(self.pin_steering, duty * 1000)
    
    def update_throttle_duty(self):
        """更新油门PWM占空比"""
        duty = self.throttle_ms_out / (1/self.pi.get_PWM_frequency(self.pin_throttle) * 1000)
        if controller_connection_check():
            self.pi.set_PWM_dutycycle(self.pin_throttle, duty * 1000)
        else:
            self.pi.set_PWM_dutycycle(self.pin_throttle, 0)
    
    def run(self):
        """主循环"""
        self.r_key_pressed = False
        operating = True
        
        try:
            while operating:
                pg.event.pump()
                
                # 检查退出事件
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        operating = False
                
                # 处理控制器重连
                self.controller_reconnection_handler()
                
                # 处理输入
                self.input_handler()
                self.steering_handler()
                self.throttle_handler()
                
                # 更新PWM
                self.update_steering_duty()
                self.update_throttle_duty()
                
                # 捕获帧（如果正在收集数据）
                self.capture_frame()
                
                # 限制循环速率
                self.clock.tick(60)
        except KeyboardInterrupt:
            print("\nProgram ended")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.collecting:
            self.stop_collecting()
        
        pg.quit()
        self.pi.set_PWM_dutycycle(self.pin_steering, 0)
        self.pi.set_PWM_dutycycle(self.pin_throttle, 0)
        self.pi.set_mode(self.pin_rec, 0)
        
        if self.cam_connected:
            try:
                self.camera.stop()
                print("Camera stopped")
            except:
                pass

# 主程序
if __name__ == "__main__":
    # 创建数据收集器，设置每5帧保存一次数据
    collector = DataCollector(frame_interval=10)
    
    print("Data collection system initialized")
    print("Press button 4 on controller or 'r' key to start/stop data collection")
    print("Use arrow keys or controller to drive")
    print("Press Ctrl+C to exit")
    
    # 运行主循环
    collector.run()
