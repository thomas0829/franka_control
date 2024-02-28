import zerorpc

from robot.real.robot_interface import RobotInterfaceServer

if __name__ == "__main__":
    robot_client = RobotInterfaceServer()
    s = zerorpc.Server(robot_client)
    s.bind("tcp://0.0.0.0:4242")
    s.run()