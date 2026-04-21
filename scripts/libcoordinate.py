import yaml

class LibCoordinate:
    def __init__(self) -> None:
        """
        初始化 LibCoordinate 类实例。

        功能说明：
            该类提供坐标转换功能。读取 main_comfig.yaml 中的坐标转换参数，据此转换坐标
        """
        # 读取配置文件
        config = yaml.load(open("main_config.yaml"), Loader=yaml.FullLoader)
        self.true_map_size_x = config["env"]["true_map_width"]
        self.true_map_size_y = config["env"]["true_map_height"]
        self.true_map_size_z = config["env"]["true_map_depth"]
        self.x_step = config["env"]["sampling_x_step"]
        self.y_step = config["env"]["sampling_y_step"]
        self.z_step = config["env"]["sampling_z_step"]

    def convert_numpy_coordinates_to_meters(self, numpy_x: int = 0, numpy_y: int = 0, numpy_z: int = 0) -> tuple[int, int, int]:
        """
        将 numpy 坐标转换为米单位坐标。

        输入参数：
            numpy_x: int
                numpy 坐标系中的 x 值。
            numpy_y: int
                numpy 坐标系中的 y 值。
            numpy_z: int
                numpy 坐标系中的 z 值。
        返回值：
            tuple[int, int, int]
                转换后的米单位坐标 (x, y, z)。
        """
        meters_x = numpy_x * self.x_step
        meters_y = numpy_y * self.y_step
        meters_z = numpy_z * self.z_step

        return meters_x, meters_y, meters_z

    def convert_meters_to_numpy_coordinates(self, meters_x: int = 0, meters_y: int = 0, meters_z: int = 0) -> tuple[int, int, int]:
        """
        将米单位坐标转换为 numpy 坐标。

        输入参数：
            meters_x: int
                米单位坐标系中的 x 值。
            meters_y: int
                米单位坐标系中的 y 值。
            meters_z: int
                米单位坐标系中的 z 值。
        返回值：
            tuple[int, int, int]
                转换后的 numpy 坐标 (x, y, z)。
        """
        numpy_x = int(meters_x / self.x_step)
        numpy_y = int(meters_y / self.y_step)
        numpy_z = int(meters_z / self.z_step)

        return numpy_x, numpy_y, numpy_z