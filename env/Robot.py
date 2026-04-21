class Robot:
    x: int
    y: int
    z: int

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def move(self, action, terrain_3d):
        """根据动作更新机器人位置，考虑地形限制

        :param action: 动作编号，0-5分别对应六个方向
        :param terrain_3d: 3D布尔数组，True表示不可通行，False表示可通行
        """
        # 定义动作对应的坐标变化
        action_map = {
            0: (-1, 0, 0),  # 向前
            1: (1, 0, 0),  # 向后
            2: (0, -1, 0),  # 向左
            3: (0, 1, 0),  # 向右
            4: (0, 0, -1),  # 向下
            5: (0, 0, 1),  # 向上
        }
        dx, dy, dz = action_map.get(action, (0, 0, 0))
        new_x = self.x + dx
        new_y = self.y + dy
        new_z = self.z + dz

        # 检查新位置是否在地图范围内且可通行
        if (0 <= new_x < terrain_3d.shape[0] and 
            0 <= new_y < terrain_3d.shape[1] and 
            0 <= new_z < terrain_3d.shape[2] and 
            not terrain_3d[new_x, new_y, new_z]):
            self.x = new_x
            self.y = new_y
            self.z = new_z

    def get_position(self):
        return (self.x, self.y, self.z)
