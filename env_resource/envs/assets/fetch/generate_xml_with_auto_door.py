BASIC_COLORS = ["0.1 0.1 0.5", "0.1 0.8 0.3", "1.0 0.9 0.0", "0.8 0.2 0.8", "1.0 0.0 0.0", "0 0 0"]

base = '''<?xml version="1.0" encoding="utf-8"?>
<mujoco>
    <compiler angle="radian" coordinate="local" meshdir="../stls/fetch" texturedir="../textures"></compiler>
    <option timestep="0.002">
        <flag warmstart="enable"></flag>
    </option>
    <include file="shared.xml"></include>
    <asset>
        {assets}
    </asset>
    <worldbody>
        <geom name="floor0" pos="0.8 0.75 0" size="3 3 3" type="plane" condim="3" material="floor_mat"></geom>
        <body name="floor0" pos="0.8 0.75 0">
        {target_sites}
        </body>
        <include file="robot.xml"></include>


        <body pos="1.3 0.75 0.2" name="table0">
            <!-- origin size 0.35 0.6 0.2 -->
            <geom size="0.35 0.3 0.2" type="box" mass="2000" material="table_mat"></geom>
        </body>

        <!-- below is the walls that I add -->
        <!-- geom name="wall0" pos="1.25 0.975 0.45" size="0.03 0.125 0.05" -->
        <!-- geom name="wall1" pos="1.25 0.525 0.45" size="0.03 0.125 0.05" -->
        <geom name="wall0" pos="1.25 0.975 0.45" size="0.03 0.125 0.05" type="box" condim="3" mass="100" material="table_mat" rgba="0.63 0.63 0.63 0.5"></geom>
        <geom name="wall1" pos="1.25 0.525 0.45" size="0.03 0.125 0.05" type="box" condim="3" mass="100" material="table_mat" rgba="0.63 0.63 0.63 0.5"></geom>

        <!-- below is the door that I add -->
        <!-- original pos="1.28 0.75 0.45" -->
        <body name="object1" pos="1.31 0.75 0.45">
            <joint axis="0 1 0" name="object1:joint" armature="0" pos="1.28 0.75 0.45" limited="true" range="-0.23 0.23" damping="4000" type="slide" springref="0" stiffness="15000"></joint>
			<geom size="0.03 0.13 0.05" type="box" condim="3" name="object1" material="stick_mat" mass="2000" friction="1 0.005 0.0001"></geom>
			<site name="object1" pos="0 0 0" size="0.02 0.02 0.02" rgba="0.5 0.2 0 0" type="sphere"></site>
		</body>

        {object_bodies}

        <light directional="true" ambient="0.2 0.2 0.2" diffuse="0.8 0.8 0.8" specular="0.3 0.3 0.3" castshadow="false" pos="0 0 4" dir="0 0 -1" name="light0"></light>
    </worldbody>
    <actuator>
        <position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:l_gripper_finger_joint" kp="30000" name="robot0:l_gripper_finger_joint" user="1"></position>
        <position ctrllimited="true" ctrlrange="0 0.2" joint="robot0:r_gripper_finger_joint" kp="30000" name="robot0:r_gripper_finger_joint" user="1"></position>
        <position ctrllimited="true" ctrlrange="0 0.2" joint="robot1:l_gripper_finger_joint" kp="30000" name="robot1:l_gripper_finger_joint" user="1"></position>
        <position ctrllimited="true" ctrlrange="0 0.2" joint="robot1:r_gripper_finger_joint" kp="30000" name="robot1:r_gripper_finger_joint" user="1"></position>
    </actuator>
</mujoco>
        '''


def generate_with_auto_door_xml(num_blocks):
    colors = BASIC_COLORS[:num_blocks]
    site_base = '<site name="target{id}" pos="0 0 0.5" size="0.02 0.02 0.02" rgba="{color} 0.3" type="sphere"></site>'
    block_base = '''<body name="object{id}" pos="0.025 0.025 0.025">
            <joint name="object{id}:joint" type="free" damping="0.01"></joint>
            <geom size="0.025 0.025 0.025" type="box" condim="3" name="object{id}" material="block{id}_mat" mass="2"></geom>
            <site name="object{id}" pos="0 0 0" size="0.02 0.02 0.02" rgba="{color} 1" type="sphere"></site>
        </body>'''
    asset_base = '<material name="block{id}_mat" specular="0" shininess="0.5" reflectance="0" rgba="{color} 1"></material>'

    sites = []
    block_bodies = []
    assets = []
    sites.append(site_base.format(**dict(id=0, color=colors[0])))
    for i in range(num_blocks):
        block_bodies.append(block_base.format(**dict(id=i, color=colors[i])))
        assets.append(asset_base.format(**dict(id=i, color=colors[i])))

    return base.format(
        **dict(assets="\n".join(assets), target_sites="\n".join(sites), object_bodies="\n".join(block_bodies)))