import abc

class FrankaBase(abc.ABC):
    def __init__(self, name, *args, **kwargs):
        self.name = name

    @abc.abstractmethod
    def get_ee_pose(self):
        """Get endeffector pose [pos (xyz), angle (euler)]"""

    @abc.abstractmethod
    def get_ee_pos(self):
        """Get endeffector position (xyz)"""

    @abc.abstractmethod
    def get_ee_angle(self):
        """Get endeffector angle (euler)"""

    @abc.abstractmethod
    def get_joint_positions(self):
        """Get robot joint positions"""

    @abc.abstractmethod
    def get_joint_velocities(self):
        """Get robot joint velocities"""

    @abc.abstractmethod
    def get_gripper_state(self):
        """Get gripper state"""

    @abc.abstractmethod
    def update_pose(self):
        """Update robot pose [pos (xyz), angle (euler)]"""

    @abc.abstractmethod
    def update_joints(self):
        """Update robot joint positions"""

    @abc.abstractmethod
    def update_gripper(self):
        """Update griper """
