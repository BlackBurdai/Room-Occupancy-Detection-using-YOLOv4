import paho.mqtt.client as mqtt
import time

class MQTT:
    def __init__(self, MQTT_BROKER):
        print("[INFO] Connecting...")
        self.client = mqtt.Client("YOLOClient")
        self.client.connect(MQTT_BROKER)
        time.sleep(2)
        print("[INFO] Connected.")

    def publish(self, TOPIC, MQTT_MSG):
        self.client.publish(TOPIC, MQTT_MSG, qos=1)
        print("[INFO] Published: " + str(MQTT_MSG) + " @TOPIC: " + TOPIC)
    
    def disconnect():
        self.client.disconnect()


# mqtt = MQTT("192.168.1.2")
# mqtt.disconnect()
# mqtt = MQTT("192.168.1.2")
# mqtt.publish("Project/RoomA/Occupancy",99)
# mqtt.publish("Project/RoomB/Occupancy",98)