import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

class UnitreeG1Env(gym.Env):
    """Custom Reinforcement Learning Environment for Unitree G1 Stability."""
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, xml_path="scene_mjx.xml"):
        super(UnitreeG1Env, self).__init__()
        
        # Load MuJoCo Model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Cache dimensions
        self.num_actuators = self.model.nu       # 29 Motors
        self.num_sensors = self.model.nsensor     # Number of custom sensors
        
        # --- DEFINE RL INTERFACES ---
        # Action Space: Raw target positions for all 29 joints (-1.0 to 1.0 normalized)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_actuators,), dtype=np.float32)
        
        # Observation Space: Pelvis tilt (3), Joint positions (29), Joint velocities (29)
        obs_shape = 3 + self.num_actuators + self.num_actuators
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

    def _get_obs(self):
        """Extract elements the RL network needs to see to maintain balance."""
        # 1. Torso tilt orientation (Roll, Pitch, Yaw orientation via freejoint quaternion/gyro)
        # For simplicity, we track the pelvis tilt rates from the first 3 elements of qvel
        torso_tilt = self.data.qvel[3:6] 
        
        # 2. Current encoder readings (Joint positions and velocities)
        joint_positions = self.data.qpos[7:] # Skip floating base coordinates (x,y,z,qx,qy,qz,qw)
        joint_velocities = self.data.qvel[6:] # Skip floating base velocities
        
        return np.concatenate([torso_tilt, joint_positions, joint_velocities]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """Resets the environment back to a stable standing initialization posture."""
        super().reset(seed=seed)
        
        # Reset physics to home keyframe
        if self.model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            mujoco.mj_resetData(self.model, self.data)
            
        observation = self._get_obs()
        info = {}
        return observation, info

    def step(self, action):
        """Executes one physics cycle based on RL Agent decisions."""
        # 1. Scale normalized actions (-1 to 1) back to raw motor limits
        # For safety/simplicity here, we scale them around the default home position
        home_posture = self.model.key_ctrl[0] if self.model.nkey > 0 else np.zeros(self.num_actuators)
        self.data.ctrl[:] = home_posture + (action * 0.2) # Allow subtle adjustments to stay upright

        # 2. Advance physics simulation
        mujoco.mj_step(self.model, self.data)
        
        # 3. Collect observations
        observation = self._get_obs()
        
        # 4. CALCULATE REWARD (Crucial for active monitoring)
        pelvis_height = self.data.qpos[2] # Z axis index
        torso_ang_vel = np.linalg.norm(self.data.qvel[3:6]) # Balance error
        
        # Reward staying tall (upright) and penalize excessive shaking/falling
        reward = float(pelvis_height - (0.1 * torso_ang_vel))
        
        # 5. Check if the robot fell down (Termination conditions)
        terminated = False
        if pelvis_height < 0.4: # Triggers if torso drops below 40 cm
            reward -= 10.0      # Penalize catastrophic fall
            terminated = True
            
        truncated = False # Can set a time limit if needed
        info = {}
        
        return observation, reward, terminated, truncated, info
if __name__ == "__main__":

    env = UnitreeG1Env(r"C:\Users\Asyncronix\Downloads\Asyncronix_Intern\Reflexive\unitree_g1\scene_mjx.xml")

    import mujoco.viewer

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        while viewer.is_running():

            mujoco.mj_step(env.model, env.data)

            viewer.sync()