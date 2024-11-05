#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

class TurtleMover(Node):
    def __init__(self):
        super().__init__('turtle_direction_listener')
        
        # Publisher for turtle movement commands
        self.pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        
        # Subscriber for direction commands
        self.create_subscription(String, '/turtle_direction', self.direction_callback, 10)
        
        self.get_logger().info("TurtleMover node has been started")

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
            self.get_logger().warn("Unknown direction: %s" % direction)
            move_cmd.linear.x = 0.0
            move_cmd.angular.z = 0.0

        return move_cmd

    def move_turtle(self, direction='forward'):
        move_cmd = self.create_twist(direction)
        

        # Publish the movement command for a short duration
        for _ in range(10):
            self.pub.publish(move_cmd)
            self.get_logger().info('Publishing: %s' % move_cmd)
            

    def direction_callback(self, msg):
        direction = msg.data
        self.get_logger().info("Received direction: %s" % direction)
        self.move_turtle(direction)

def main(args=None):
    rclpy.init(args=args)
    
    turtle_mover = TurtleMover()
    
    try:
        rclpy.spin(turtle_mover)
    except KeyboardInterrupt:
        pass
    finally:
        turtle_mover.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

