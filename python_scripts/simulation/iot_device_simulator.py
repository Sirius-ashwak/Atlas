"""
IoT Device Simulator for MQTT-based Hybrid Simulation

Simulates multiple IoT devices publishing telemetry data to MQTT broker.
Each device publishes metrics like CPU, memory, energy, latency, etc.

Usage:
    python iot_device_simulator.py --num-devices 10 --interval 5
    python iot_device_simulator.py --broker localhost --port 1883
"""

import argparse
import json
import random
import time
from datetime import datetime
from typing import Dict, List
import sys

try:
    import paho.mqtt.client as mqtt
except ImportError:
    print("❌ Error: paho-mqtt not installed. Install with:")
    print("   pip install paho-mqtt")
    sys.exit(1)


class IoTDevice:
    """Simulates a single IoT device with realistic behavior."""
    
    DEVICE_TYPES = {
        0: "sensor",      # Low resource devices
        1: "fog_node",    # Medium resource edge servers
        2: "cloud_node"   # High resource cloud servers
    }
    
    def __init__(self, device_id: int, device_type: int):
        self.device_id = device_id
        self.device_type = device_type
        self.type_name = self.DEVICE_TYPES[device_type]
        
        # Initialize base metrics based on device type
        self._init_base_metrics()
        
        # State tracking for realistic simulation
        self.cpu_trend = random.choice([-0.01, 0.01])
        self.mem_trend = random.choice([-0.01, 0.01])
        self.last_update = time.time()
    
    def _init_base_metrics(self):
        """Initialize metrics based on device type."""
        if self.device_type == 0:  # Sensor
            self.cpu_util = random.uniform(0.1, 0.4)
            self.mem_util = random.uniform(0.2, 0.5)
            self.energy = random.uniform(10, 50)
            self.latency = random.uniform(10, 100)
            self.bandwidth = random.uniform(10, 100)
            self.queue_len = random.randint(0, 5)
        elif self.device_type == 1:  # Fog Node
            self.cpu_util = random.uniform(0.3, 0.7)
            self.mem_util = random.uniform(0.4, 0.7)
            self.energy = random.uniform(50, 100)
            self.latency = random.uniform(5, 30)
            self.bandwidth = random.uniform(100, 300)
            self.queue_len = random.randint(0, 10)
        else:  # Cloud Node
            self.cpu_util = random.uniform(0.2, 0.6)
            self.mem_util = random.uniform(0.3, 0.6)
            self.energy = random.uniform(100, 200)
            self.latency = random.uniform(1, 10)
            self.bandwidth = random.uniform(300, 1000)
            self.queue_len = random.randint(0, 15)
    
    def update_metrics(self):
        """Update metrics with realistic temporal behavior."""
        # CPU utilization with trending
        self.cpu_util += self.cpu_trend + random.uniform(-0.05, 0.05)
        self.cpu_util = max(0.1, min(0.95, self.cpu_util))
        
        # Reverse trend at boundaries
        if self.cpu_util > 0.9 or self.cpu_util < 0.15:
            self.cpu_trend *= -1
        
        # Memory utilization with trending
        self.mem_util += self.mem_trend + random.uniform(-0.03, 0.03)
        self.mem_util = max(0.1, min(0.90, self.mem_util))
        
        if self.mem_util > 0.85 or self.mem_util < 0.20:
            self.mem_trend *= -1
        
        # Energy consumption (correlated with CPU)
        energy_base = {0: 30, 1: 75, 2: 150}[self.device_type]
        self.energy = energy_base * (0.5 + 0.5 * self.cpu_util)
        
        # Queue length changes
        queue_change = random.randint(-2, 3)
        max_queue = {0: 8, 1: 15, 2: 20}[self.device_type]
        self.queue_len = max(0, min(max_queue, self.queue_len + queue_change))
        
        # Latency with some jitter
        self.latency *= random.uniform(0.9, 1.1)
        
        # Bandwidth with seasonal variation
        self.bandwidth *= random.uniform(0.95, 1.05)
    
    def get_telemetry(self) -> Dict:
        """Get current telemetry data."""
        return {
            "device_id": self.device_id,
            "device_type": self.device_type,
            "type_name": self.type_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "cpu_util": round(self.cpu_util, 4),
                "mem_util": round(self.mem_util, 4),
                "energy": round(self.energy, 2),
                "latency": round(self.latency, 2),
                "bandwidth": round(self.bandwidth, 2),
                "queue_len": self.queue_len
            }
        }


class IoTSimulator:
    """Manages multiple IoT devices and MQTT communication."""
    
    def __init__(self, broker: str, port: int, num_devices: int, 
                 interval: float, topic_prefix: str = "iot/devices"):
        self.broker = broker
        self.port = port
        self.num_devices = num_devices
        self.interval = interval
        self.topic_prefix = topic_prefix
        
        # Create devices with realistic distribution
        self.devices: List[IoTDevice] = []
        self._create_devices()
        
        # MQTT client
        self.client = mqtt.Client(client_id=f"iot_simulator_{random.randint(1000, 9999)}")
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish
        
        self.connected = False
        self.publish_count = 0
    
    def _create_devices(self):
        """Create devices with realistic type distribution."""
        # Distribution: 50% sensors, 30% fog, 20% cloud
        num_sensors = int(self.num_devices * 0.5)
        num_fog = int(self.num_devices * 0.3)
        num_cloud = self.num_devices - num_sensors - num_fog
        
        device_id = 0
        
        # Create sensors
        for _ in range(num_sensors):
            self.devices.append(IoTDevice(device_id, 0))
            device_id += 1
        
        # Create fog nodes
        for _ in range(num_fog):
            self.devices.append(IoTDevice(device_id, 1))
            device_id += 1
        
        # Create cloud nodes
        for _ in range(num_cloud):
            self.devices.append(IoTDevice(device_id, 2))
            device_id += 1
        
        print(f"✅ Created {len(self.devices)} devices: {num_sensors} sensors, "
              f"{num_fog} fog nodes, {num_cloud} cloud nodes")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            self.connected = True
            print(f"✅ Connected to MQTT broker at {self.broker}:{self.port}")
        else:
            print(f"❌ Connection failed with code {rc}")
            self.connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker."""
        self.connected = False
        if rc != 0:
            print(f"⚠️ Unexpected disconnection (code {rc})")
    
    def _on_publish(self, client, userdata, mid):
        """Callback when message is published."""
        self.publish_count += 1
    
    def connect(self):
        """Connect to MQTT broker."""
        try:
            print(f"🔌 Connecting to MQTT broker at {self.broker}:{self.port}...")
            self.client.connect(self.broker, self.port, keepalive=60)
            self.client.loop_start()
            
            # Wait for connection
            timeout = 5
            start_time = time.time()
            while not self.connected and (time.time() - start_time) < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                raise ConnectionError("Failed to connect within timeout")
            
            return True
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return False
    
    def publish_telemetry(self):
        """Publish telemetry from all devices."""
        if not self.connected:
            print("⚠️ Not connected to broker")
            return
        
        batch_data = {
            "timestamp": datetime.now().isoformat(),
            "num_devices": len(self.devices),
            "devices": []
        }
        
        for device in self.devices:
            # Update device metrics
            device.update_metrics()
            
            # Get telemetry
            telemetry = device.get_telemetry()
            batch_data["devices"].append(telemetry)
            
            # Publish individual device data
            topic = f"{self.topic_prefix}/{device.device_id}"
            payload = json.dumps(telemetry)
            self.client.publish(topic, payload, qos=1)
        
        # Publish batch data
        batch_topic = f"{self.topic_prefix}/batch"
        batch_payload = json.dumps(batch_data)
        self.client.publish(batch_topic, batch_payload, qos=1)
        
        print(f"📡 Published telemetry from {len(self.devices)} devices "
              f"(total: {self.publish_count} messages)")
    
    def run(self):
        """Run continuous simulation."""
        print("\n" + "="*60)
        print("🚀 IoT Device Simulator Started")
        print("="*60)
        print(f"Broker: {self.broker}:{self.port}")
        print(f"Devices: {self.num_devices}")
        print(f"Update Interval: {self.interval}s")
        print(f"Topic Prefix: {self.topic_prefix}")
        print("="*60 + "\n")
        
        if not self.connect():
            print("❌ Failed to connect to broker. Exiting.")
            return
        
        try:
            iteration = 0
            while True:
                iteration += 1
                print(f"\n📊 Iteration {iteration} - {datetime.now().strftime('%H:%M:%S')}")
                
                # Publish telemetry
                self.publish_telemetry()
                
                # Wait for next interval
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            print("\n\n⚠️ Simulation interrupted by user")
        finally:
            self.disconnect()
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        print("\n🔌 Disconnecting from broker...")
        self.client.loop_stop()
        self.client.disconnect()
        print(f"✅ Published {self.publish_count} total messages")
        print("👋 Simulator stopped")


def main():
    parser = argparse.ArgumentParser(
        description="IoT Device Simulator with MQTT Publishing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 10 devices, 5 second interval
  python iot_device_simulator.py
  
  # Custom settings
  python iot_device_simulator.py --num-devices 20 --interval 3
  
  # Remote broker
  python iot_device_simulator.py --broker 192.168.1.100 --port 1883
        """
    )
    
    parser.add_argument(
        "--broker", "-b",
        type=str,
        default="localhost",
        help="MQTT broker address (default: localhost)"
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=1883,
        help="MQTT broker port (default: 1883)"
    )
    
    parser.add_argument(
        "--num-devices", "-n",
        type=int,
        default=10,
        help="Number of IoT devices to simulate (default: 10)"
    )
    
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=5.0,
        help="Telemetry publish interval in seconds (default: 5.0)"
    )
    
    parser.add_argument(
        "--topic-prefix", "-t",
        type=str,
        default="iot/devices",
        help="MQTT topic prefix (default: iot/devices)"
    )
    
    args = parser.parse_args()
    
    # Create and run simulator
    simulator = IoTSimulator(
        broker=args.broker,
        port=args.port,
        num_devices=args.num_devices,
        interval=args.interval,
        topic_prefix=args.topic_prefix
    )
    
    simulator.run()


if __name__ == "__main__":
    main()
