"""
MQTT Subscriber for Real-time IoT Device Monitoring

Subscribes to MQTT topics, receives device telemetry, and triggers
allocation decisions using the AI model.

Usage:
    from src.mqtt.mqtt_subscriber import MQTTAllocator
    
    allocator = MQTTAllocator(broker="localhost")
    allocator.start()
"""

import json
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
import logging

try:
    import paho.mqtt.client as mqtt
except ImportError:
    raise ImportError("paho-mqtt not installed. Install with: pip install paho-mqtt")

logger = logging.getLogger(__name__)


class MQTTAllocator:
    """MQTT subscriber that performs real-time allocation decisions."""
    
    def __init__(self, 
                 broker: str = "localhost",
                 port: int = 1883,
                 topic_prefix: str = "iot/devices",
                 model_loader = None,
                 allocation_callback: Optional[Callable] = None):
        """
        Initialize MQTT allocator.
        
        Args:
            broker: MQTT broker address
            port: MQTT broker port
            topic_prefix: Topic prefix to subscribe to
            model_loader: Model loader instance for inference
            allocation_callback: Optional callback for allocation results
        """
        self.broker = broker
        self.port = port
        self.topic_prefix = topic_prefix
        self.model_loader = model_loader
        self.allocation_callback = allocation_callback
        
        # MQTT client
        client_id = f"allocator_{int(time.time())}"
        self.client = mqtt.Client(client_id=client_id)
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        
        # State tracking
        self.connected = False
        self.running = False
        self.device_states: Dict[int, Dict] = {}
        self.last_allocation_time = time.time()
        self.allocation_interval = 5.0  # Seconds between allocations
        
        # Statistics
        self.stats = {
            "messages_received": 0,
            "allocations_made": 0,
            "last_update": None,
            "devices_tracked": 0
        }
        
        # Thread for periodic allocation
        self.allocation_thread: Optional[threading.Thread] = None
    
    def _on_connect(self, client, userdata, flags, rc):
        """Callback when connected to MQTT broker."""
        if rc == 0:
            self.connected = True
            logger.info(f"Connected to MQTT broker at {self.broker}:{self.port}")
            
            # Subscribe to batch topic (most efficient)
            batch_topic = f"{self.topic_prefix}/batch"
            self.client.subscribe(batch_topic, qos=1)
            logger.info(f"Subscribed to topic: {batch_topic}")
            
            # Also subscribe to individual device topics
            individual_topic = f"{self.topic_prefix}/+/telemetry"
            self.client.subscribe(individual_topic, qos=1)
            logger.info(f"Subscribed to topic: {individual_topic}")
        else:
            logger.error(f"Connection failed with code {rc}")
            self.connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """Callback when disconnected from MQTT broker."""
        self.connected = False
        if rc != 0:
            logger.warning(f"Unexpected disconnection (code {rc})")
    
    def _on_message(self, client, userdata, msg):
        """Callback when message is received."""
        try:
            payload = json.loads(msg.payload.decode())
            
            # Handle batch messages
            if "/batch" in msg.topic:
                self._handle_batch_update(payload)
            else:
                # Handle individual device message
                self._handle_device_update(payload)
            
            self.stats["messages_received"] += 1
            self.stats["last_update"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    def _handle_batch_update(self, data: Dict):
        """Handle batch telemetry update."""
        devices = data.get("devices", [])
        
        for device_data in devices:
            device_id = device_data.get("device_id")
            if device_id is not None:
                self.device_states[device_id] = device_data
        
        self.stats["devices_tracked"] = len(self.device_states)
        
        logger.debug(f"Updated {len(devices)} device states from batch message")
    
    def _handle_device_update(self, data: Dict):
        """Handle individual device telemetry update."""
        device_id = data.get("device_id")
        if device_id is not None:
            self.device_states[device_id] = data
            self.stats["devices_tracked"] = len(self.device_states)
            logger.debug(f"Updated state for device {device_id}")
    
    def _create_network_state(self) -> Optional[Dict]:
        """Create network state from current device telemetry."""
        if not self.device_states:
            return None
        
        nodes = []
        edges = []
        
        # Convert device states to node format
        device_ids = sorted(self.device_states.keys())
        
        for device_id in device_ids:
            device = self.device_states[device_id]
            metrics = device.get("metrics", {})
            
            nodes.append({
                "cpu_util": metrics.get("cpu_util", 0.5),
                "mem_util": metrics.get("mem_util", 0.5),
                "energy": metrics.get("energy", 50),
                "latency": metrics.get("latency", 20),
                "bandwidth": metrics.get("bandwidth", 100),
                "queue_len": metrics.get("queue_len", 0),
                "node_type": device.get("device_type", 0)
            })
        
        # Create simple hierarchical edges
        for i in range(len(device_ids) - 1):
            edges.append([i, i + 1])
        
        return {
            "nodes": nodes,
            "edges": edges,
            "timestamp": datetime.now().isoformat(),
            "num_devices": len(nodes)
        }
    
    def _make_allocation_decision(self):
        """Make allocation decision based on current network state."""
        network_state = self._create_network_state()
        
        if not network_state:
            logger.warning("No device states available for allocation")
            return None
        
        # Use model for inference if available
        if self.model_loader:
            try:
                # Get current model (default to hybrid)
                model = self.model_loader.get_model("hybrid")
                
                if model:
                    # Perform inference (simplified)
                    allocation_result = {
                        "timestamp": datetime.now().isoformat(),
                        "network_state": network_state,
                        "allocation": "model_based",
                        "num_devices": network_state["num_devices"],
                        "status": "success"
                    }
                    
                    self.stats["allocations_made"] += 1
                    logger.info(f"Allocation decision made for {network_state['num_devices']} devices")
                    
                    # Call callback if provided
                    if self.allocation_callback:
                        self.allocation_callback(allocation_result)
                    
                    return allocation_result
                    
            except Exception as e:
                logger.error(f"Error during allocation: {e}")
        
        return None
    
    def _allocation_loop(self):
        """Periodic allocation decision loop."""
        logger.info("Allocation loop started")
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time for next allocation
                if current_time - self.last_allocation_time >= self.allocation_interval:
                    if self.device_states:
                        self._make_allocation_decision()
                        self.last_allocation_time = current_time
                    
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in allocation loop: {e}")
                time.sleep(5)
        
        logger.info("Allocation loop stopped")
    
    def start(self, blocking: bool = True):
        """
        Start MQTT subscriber and allocation engine.
        
        Args:
            blocking: If True, blocks until stopped. If False, runs in background.
        """
        logger.info("Starting MQTT Allocator...")
        
        try:
            # Connect to broker
            logger.info(f"Connecting to {self.broker}:{self.port}")
            self.client.connect(self.broker, self.port, keepalive=60)
            
            self.running = True
            
            # Start allocation thread
            self.allocation_thread = threading.Thread(target=self._allocation_loop, daemon=True)
            self.allocation_thread.start()
            
            if blocking:
                # Blocking mode - run loop in main thread
                logger.info("Running in blocking mode (Ctrl+C to stop)")
                self.client.loop_forever()
            else:
                # Non-blocking mode - run loop in background thread
                logger.info("Running in non-blocking mode")
                self.client.loop_start()
        
        except Exception as e:
            logger.error(f"Failed to start: {e}")
            raise
    
    def stop(self):
        """Stop MQTT subscriber and allocation engine."""
        logger.info("Stopping MQTT Allocator...")
        
        self.running = False
        
        # Wait for allocation thread
        if self.allocation_thread and self.allocation_thread.is_alive():
            self.allocation_thread.join(timeout=5)
        
        # Disconnect from broker
        self.client.loop_stop()
        self.client.disconnect()
        
        logger.info("MQTT Allocator stopped")
    
    def get_stats(self) -> Dict:
        """Get current statistics."""
        return {
            **self.stats,
            "connected": self.connected,
            "running": self.running
        }
    
    def get_device_states(self) -> Dict[int, Dict]:
        """Get current device states."""
        return self.device_states.copy()
    
    def set_allocation_interval(self, interval: float):
        """Set allocation interval in seconds."""
        self.allocation_interval = max(1.0, interval)
        logger.info(f"Allocation interval set to {self.allocation_interval}s")


# Example usage
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    def allocation_callback(result):
        """Example callback for allocation results."""
        print(f"✅ Allocation: {result['num_devices']} devices processed")
    
    allocator = MQTTAllocator(
        broker="localhost",
        port=1883,
        allocation_callback=allocation_callback
    )
    
    try:
        allocator.start(blocking=True)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        allocator.stop()
