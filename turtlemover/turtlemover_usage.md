##Turtlemover

Turtlemover listens to topic /turtle_direction for messages left, right, forward, backward and moves the robot in
the direction it receives.

Use (with turtlesim) by creating an environment with turlesim, launching turtlesim and turtlemover.py. 
When a message is sent to the /turtle_direction topic, the turtle should move.
Messages can be sent for example with
```
ros2 topic pub /turtle_direction std_msgs/msg/String "data: 'forward'"
```
