import random

import carla
import numpy as np

PEDESTRIAN_TYPE = {
    "cautious": 0.3,
    "normal": 0.5,
    "aggressive": 1.0,
}

TO_TARGET_THRESHOLD = 1.0
RANGE_BUFFER = 0.1
SPEED_FACTOR = 3.0
PEDESTRIAN_SPEED = 5


class PedestrianAgent:
    def __init__(self, walker, behavior="normal", plan=False):
        self.walker = walker
        self.ai_controller = None
        self.behavior = behavior
        self.plan = plan
        self.start_walking = False
        self.destination = None
        self.des_norm_vector = None
        self.speed = None
        self.ego_agent = None  # This is used only for adaptive control
        self.factor = 1.0
        self.ego_straight_vector = None
        
        # [TO_TARGET_THRESHOLD - RANGE_BUFFER, TO_TARGET_THRESHOLD + RANGE_BUFFER]
        self.to_finish = ((random.random() * 2) - 1) * RANGE_BUFFER + TO_TARGET_THRESHOLD

    def set_controller(self, ai_controller):
        self.ai_controller = ai_controller

    def set_ego_agent(self, ego_agent, factor=1.0):
        self.ego_agent = ego_agent
        self.factor = factor

    def set_ego_vector(self, ego_waypoint):
        next_waypoint = ego_waypoint.next(1)[0]
        diff_vector = np.array(
            [
                next_waypoint.transform.location.x - ego_waypoint.transform.location.x,
                next_waypoint.transform.location.y - ego_waypoint.transform.location.y,
            ]
        )
        self.ego_straight_vector = diff_vector / np.linalg.norm(diff_vector)

    def set_destination(self, destination):
        self.destination = destination
        direction = self.destination - self.walker.get_location()
        norm = np.linalg.norm([direction.x, direction.y])
        if norm < 1e-3:
            self.des_norm_vector = carla.Vector3D(1, 0, 0)
        else:
            self.des_norm_vector = carla.Vector3D(direction.x / norm, direction.y / norm, 0)

    def check_finish(self):
        current_location = self.walker.get_location()
        if self.destination is not None:
            return self.destination.distance(current_location) < self.to_finish

    def _normal_step(self, speed):
        self.walker.apply_control(
            carla.WalkerControl(
                direction=self.des_norm_vector,
                speed=speed,
            )
        )

    def _plan_step(self, move_dis_threshold=0.35, dynamic_range=0.25):
        ego_location = self.ego_agent.get_location()
        walker_location = self.walker.get_location()
        
        # [move_dis_threshold, move_dis_threshold + dynamic_range]
        move_dis_threshold = random.random() * dynamic_range + move_dis_threshold  # 0.35 ~ 0.6

        vec_to_ego = np.array(
            [ego_location.x - walker_location.x, ego_location.y - walker_location.y]
        )
        vec_to_des = np.array(
            [self.destination.x - walker_location.x, self.destination.y - walker_location.y]
        )
        vec_to_ego = vec_to_ego / np.linalg.norm(vec_to_ego)
        vec_to_des = vec_to_des / np.linalg.norm(vec_to_des)

        cos = np.abs(np.dot(vec_to_ego, vec_to_des))

        if not self.start_walking:
            self.start_walking = cos >= move_dis_threshold
            if not self.start_walking:
                return
        ego_agent_throttle = self.ego_agent.get_control().throttle
        if ego_agent_throttle < 1e-3:
            return
        self._normal_step(speed=PEDESTRIAN_SPEED)

    def run_step(self):
        if self.destination is None or not self.walker.is_alive or not self.ego_agent.is_alive:
            if self.ai_controller is not None:
                self.ai_controller.stop()
            return True
        if self.check_finish():
            if self.ai_controller is None:
                self.walker.apply_control(carla.WalkerControl())
                self.destination = None
            else:
                self.ai_controller.stop()
            return True

        if self.ai_controller is not None:
            return False

        if self.plan:
            self._plan_step()
        else:
            self._normal_step(
                speed=(1 + random.random()) * PEDESTRIAN_TYPE[self.behavior],
            )
        return False
