#!/usr/bin/env python3
import mujoco as mj
import numpy as np
from mujoco_base import MuJoCoBase
from mujoco.glfw import glfw
import rospy
import rospkg
from std_msgs.msg import Float32MultiArray,Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

import time

class HumanoidSim(MuJoCoBase):
  def __init__(self, xml_path):
    super().__init__(xml_path)
    self.simend = 1000.0
    # print('Total number of DoFs in the model:', self.model.nv)
    # print('Generalized positions:', self.data.qpos)  
    # print('Generalized velocities:', self.data.qvel)
    # print('Actuator forces:', self.data.qfrc_actuator)
    # print('Actoator controls:', self.data.ctrl)
    # mj.set_mjcb_control(self.controller)
    # * Set subscriber and publisher

    # initial_qpos
    init_l_leg_qpos = np.array([0, 0, -0.41, 1, -0.59, 0])
    init_r_leg_qpos = init_l_leg_qpos.copy()
    init_leg_qpos = np.concatenate([init_l_leg_qpos, init_l_leg_qpos])

    init_l_arm_qpos = np.array([0, 0.3, 0, -0.5, 0, 0, 0])
    init_r_arm_qpos = np.array([0, -0.3, 0, -0.5, 0, 0, 0])
    init_arm_qpos = np.concatenate([init_l_arm_qpos, init_r_arm_qpos])
    init_other_qpos = np.concatenate([np.zeros(6), init_arm_qpos])

    init_height = 0.86

    # initialize target joint position, velocity, and torque
    # for legs
    self.n_leg = 12
    targetPosLeg = init_leg_qpos.copy()
    targetVelLeg = np.zeros(self.n_leg)
    targetTorqueLeg = np.zeros(self.n_leg)
    targetKpLeg = np.ones(self.n_leg) * 300
    targetKdLeg = np.ones(self.n_leg) * 20

    # for other joints
    self.n_other = 20
    targetPosOther = init_other_qpos.copy()
    targetVelOther = np.zeros(self.n_other)
    targetTorqueOther = np.zeros(self.n_other)
    targetKpOther = np.ones(self.n_other) * 1
    targetKdOther = np.ones(self.n_other) * 0

    # all joints
    # self.n = self.n_leg + self.n_other # 32
    # self.targetPos = np.concatenate([targetPosLeg, targetPosOther]) # 32
    # self.targetVel = np.concatenate([targetVelLeg, targetVelOther])
    # self.targetTorque = np.concatenate([targetTorqueLeg, targetTorqueOther])
    # self.targetKp = np.concatenate([targetKpLeg, targetKpOther])
    # self.targetKd = np.concatenate([targetKdLeg, targetKdOther])

    self.targetPos = targetPosLeg.copy()
    self.targetVel = targetVelLeg.copy()
    self.targetTorque = targetTorqueLeg.copy()
    self.targetKp = targetKpLeg.copy()
    self.targetKd = targetKdLeg.copy()

    # index
    self.leg_pos_index = slice(7, 19)
    self.other_pos_index = slice(19, 39)
    self.leg_vel_index = slice(6, 18)
    self.other_vel_index = slice(18, 38)

    self.pubJoints = rospy.Publisher('/jointsPosVel', Float32MultiArray, queue_size=10)
    self.pubOdom = rospy.Publisher('/ground_truth/state', Odometry, queue_size=10)
    self.pubImu = rospy.Publisher('/imu', Imu, queue_size=10)
    self.pubRealTorque = rospy.Publisher('/realTorque', Float32MultiArray, queue_size=10)

    rospy.Subscriber("/targetTorque", Float32MultiArray, self.targetTorqueCallback) 
    rospy.Subscriber("/targetPos", Float32MultiArray, self.targetPosCallback) 
    rospy.Subscriber("/targetVel", Float32MultiArray, self.targetVelCallback)
    rospy.Subscriber("/targetKp", Float32MultiArray, self.targetKpCallback)
    rospy.Subscriber("/targetKd", Float32MultiArray, self.targetKdCallback)
    #set the initial joint position

    self.data.qpos[:3] = np.array([0, 0, init_height])
    self.data.qpos[self.leg_pos_index] = init_leg_qpos.copy()
    # self.data.qpos[self.other_pos_index] = init_other_qpos.copy()
    # self.data.qvel[:3] = np.array([0, 0, 0])
    # self.data.qvel[-12:] = np.zeros(12)

    # * show the model
    mj.mj_step(self.model, self.data)
    # enable contact force visualization
    self.opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        self.window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    # Update scene and render
    mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                        mj.mjtCatBit.mjCAT_ALL.value, self.scene)
    mj.mjr_render(viewport, self.scene, self.context)
    

  def targetTorqueCallback(self, data):
    self.targetTorque[:self.n_leg] = data.data

  def targetPosCallback(self, data):
    self.targetPos[:self.n_leg] = data.data

  def targetVelCallback(self, data):
    self.targetVel[:self.n_leg] = data.data 

  def targetKpCallback(self, data):
    self.targetKp[:self.n_leg] = data.data

  def targetKdCallback(self, data):
    self.targetKd[:self.n_leg] = data.data

  def reset(self):
    # Set camera configuration
    self.cam.azimuth = 89.608063
    self.cam.elevation = -11.588379
    self.cam.distance = 5.0
    self.cam.lookat = np.array([0.0, 0.0, 1.5])

  # def controller(self, model, data):
  #   self.data.ctrl[0] = 100
  #   pass

  def simulate(self):
    publish_time = self.data.time
    torque_publish_time = self.data.time
    while not glfw.window_should_close(self.window):
      simstart = self.data.time
      while (self.data.time - simstart <= 1.0/60 and not self.pause_flag):

        # MIT control
        # print(f'targetTorque: {self.targetTorque}')
        # print(f'targetPos: {self.targetPos}')
        # print(f'targetVel: {self.targetVel}')
        # exit()        
        self.data.ctrl[:] = self.targetTorque + self.targetKp * (self.targetPos - self.data.qpos[7:]) + self.targetKd * (self.targetVel - self.data.qvel[6:])
        # print(self.data.ctrl)
        # Step simulation environment
        mj.mj_step(self.model, self.data)
        if (self.data.time - publish_time >= 1.0 / 500.0):
          # * Publish joint positions and velocities
          jointsPosVel = Float32MultiArray()
          # get last 12 element of qpos and qvel
          qp = self.data.qpos[self.leg_pos_index].copy()
          # qv = self.data.qvel[self.leg_vel_index].copy()
          jointsPosVel.data = np.concatenate((qp,qv))

          self.pubJoints.publish(jointsPosVel)
          # * Publish body pose
          bodyOdom = Odometry()
          pos = self.data.sensor('BodyPos').data.copy()
          ori = self.data.sensor('BodyQuat').data.copy()
          vel = self.data.qvel[:3].copy()
          angVel = self.data.sensor('BodyGyro').data.copy()

          bodyOdom.header.stamp = rospy.Time.now()
          bodyOdom.pose.pose.position.x = pos[0]
          bodyOdom.pose.pose.position.y = pos[1]
          bodyOdom.pose.pose.position.z = pos[2]
          bodyOdom.pose.pose.orientation.x = ori[1]
          bodyOdom.pose.pose.orientation.y = ori[2]
          bodyOdom.pose.pose.orientation.z = ori[3]
          bodyOdom.pose.pose.orientation.w = ori[0]
          bodyOdom.twist.twist.linear.x = vel[0]
          bodyOdom.twist.twist.linear.y = vel[1]
          bodyOdom.twist.twist.linear.z = vel[2]
          bodyOdom.twist.twist.angular.x = angVel[0]
          bodyOdom.twist.twist.angular.y = angVel[1]
          bodyOdom.twist.twist.angular.z = angVel[2]
          self.pubOdom.publish(bodyOdom)

          bodyImu = Imu()
          acc = self.data.sensor('BodyAcc').data.copy()
          bodyImu.header.stamp = rospy.Time.now()
          bodyImu.angular_velocity.x = angVel[0]
          bodyImu.angular_velocity.y = angVel[1]
          bodyImu.angular_velocity.z = angVel[2]
          bodyImu.linear_acceleration.x = acc[0]
          bodyImu.linear_acceleration.y = acc[1]
          bodyImu.linear_acceleration.z = acc[2]
          bodyImu.orientation.x = ori[1]
          bodyImu.orientation.y = ori[2]
          bodyImu.orientation.z = ori[3]
          bodyImu.orientation.w = ori[0]
          bodyImu.orientation_covariance = [0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0]
          bodyImu.angular_velocity_covariance = [0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0]
          bodyImu.linear_acceleration_covariance = [0.0, 0, 0, 0, 0.0, 0, 0, 0, 0.0]
          self.pubImu.publish(bodyImu)

          publish_time = self.data.time

      if (self.data.time - torque_publish_time >= 1.0 / 40.0):
        targetTorque = Float32MultiArray()
        targetTorque.data = self.data.ctrl[:self.n_leg]
        self.pubRealTorque.publish(targetTorque)
        torque_publish_time = self.data.time

      if self.data.time >= self.simend:
          break
      if self.pause_flag:
        # publish the state even if the simulation is paused
        # * Publish joint positions and velocities
        jointsPosVel = Float32MultiArray()
        # get last 12 element of qpos and qvel
        qp = self.data.qpos[self.leg_pos_index].copy()
        qv = np.zeros(12)
        jointsPosVel.data = np.concatenate((qp,qv))

        self.pubJoints.publish(jointsPosVel)
        # * Publish body pose
        bodyOdom = Odometry()
        pos = self.data.sensor('BodyPos').data.copy()
        ori = self.data.sensor('BodyQuat').data.copy()
        vel = self.data.qvel[:3].copy()
        angVel = self.data.sensor('BodyGyro').data.copy()
        bodyOdom.header.stamp = rospy.Time.now()
        bodyOdom.pose.pose.position.x = pos[0]
        bodyOdom.pose.pose.position.y = pos[1]
        bodyOdom.pose.pose.position.z = pos[2]
        bodyOdom.pose.pose.orientation.x = ori[1]
        bodyOdom.pose.pose.orientation.y = ori[2]
        bodyOdom.pose.pose.orientation.z = ori[3]
        bodyOdom.pose.pose.orientation.w = ori[0]
        bodyOdom.twist.twist.linear.x = 0
        bodyOdom.twist.twist.linear.y = 0
        bodyOdom.twist.twist.linear.z = 0
        bodyOdom.twist.twist.angular.x = 0
        bodyOdom.twist.twist.angular.y = 0
        bodyOdom.twist.twist.angular.z = 0
        self.pubOdom.publish(bodyOdom)

        bodyImu = Imu()
        bodyImu.header.stamp = rospy.Time.now()
        bodyImu.angular_velocity.x = 0
        bodyImu.angular_velocity.y = 0
        bodyImu.angular_velocity.z = 0
        bodyImu.linear_acceleration.x = 0
        bodyImu.linear_acceleration.y = 0
        bodyImu.linear_acceleration.z = 9.81
        bodyImu.orientation.x = ori[1]
        bodyImu.orientation.y = ori[2]
        bodyImu.orientation.z = ori[3]
        bodyImu.orientation.w = ori[0]
        self.pubImu.publish(bodyImu)

      # get framebuffer viewport
      viewport_width, viewport_height = glfw.get_framebuffer_size(
          self.window)
      viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

      # Update scene and render
      mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                          mj.mjtCatBit.mjCAT_ALL.value, self.scene)
      mj.mjr_render(viewport, self.scene, self.context)

      # swap OpenGL buffers (blocking call due to v-sync)
      glfw.swap_buffers(self.window)

      # process pending GUI events, call GLFW callbacks
      glfw.poll_events()

    glfw.terminate()

def main():
    # ros init
    rospy.init_node('hector_sim', anonymous=True)

    # get xml path
    rospack = rospkg.RosPack()
    rospack.list()
    # hector_desc_path = rospack.get_path('humanoid_legged_description')
    # xml_path = hector_desc_path + "/mjcf/humanoid_legged.xml"
    gr1t1_path = rospack.get_path('gr1t1_description')
    xml_path = gr1t1_path + "/mjcf/scene.xml"
    sim = HumanoidSim(xml_path)
    sim.reset()
    sim.simulate()

if __name__ == "__main__":
    main()
