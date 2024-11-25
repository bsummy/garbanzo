#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import socket

"""
connects to llminstruct.py by socket to avoid version problems
"""
 
class TurtleAIController(Node):
    def __init__(self):
        super().__init__('turtle_ai_controller')
 
        # ROS 2 Publisher for turtle movement commands
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
 
        # Socket Setup
        self.HOST = "127.0.0.1"  # AI Backend Host
        self.PORT = 65432        # AI Backend Port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.HOST, self.PORT))
        self.get_logger().info("Connected to AI Backend")
 
    def create_twist(self, direction):
        move_cmd = Twist()
 
        if direction == 'forward':
            move_cmd.linear.x = 1.0
            move_cmd.angular.z = 0.0
        elif direction == 'backward':
            move_cmd.linear.x = -1.0
            move_cmd.angular.z = 0.0
        elif direction == 'left':
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 1.0
        elif direction == 'right':
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = -1.0
        else:
            self.get_logger().warn(f"Unknown direction: {direction}")
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.0
 
        return move_cmd
 
    def send_twist_commands(self, commands):
        for command in commands:
            command = command.strip()
            twist = self.create_twist(command)
            for _ in range(10):  # Publish the command multiple times for smooth motion
                self.pub.publish(twist)
            self.get_logger().info(f"Executing: {command}")
 
    def run(self):
        while rclpy.ok():
            try:
                # Get user instruction
                instruction = input("Enter instructions (or type 'exit' to quit): ")
                if instruction.lower() == 'exit':
                    self.socket.sendall(b'exit')
                    self.get_logger().info("Exiting...")
                    break
                print("sending instruction")
                # Send instruction to AI backend
                self.socket.sendall(instruction.encode('utf-8'))
 
                # Receive response from AI backend
                response = self.socket.recv(1024).decode('utf-8')
                self.get_logger().info(f"AI Response: {response}")
 
                # Split the response
                commands = response.split(",")
                self.send_twist_commands(commands)
 
            except KeyboardInterrupt:
                break
 
        # Cleanup
        self.socket.close()
        self.get_logger().info("Disconnected from AI Backend")
 
def main(args=None):
    rclpy.init(args=args)
    controller = TurtleAIController()
 
    try:
        controller.run()
    except KeyboardInterrupt:
        pass
    finally:
        controller.destroy_node()
        rclpy.shutdown()
 
if __name__ == '__main__':
    main()
