import os
from typing import Dict, List

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.colors import Colormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.legend_handler import HandlerBase
import matplotlib.image as mpimg
from matplotlib.image import BboxImage
from matplotlib.transforms import TransformedBbox, Bbox

from multi_agent_power_allocation import BASE_DIR
from multi_agent_power_allocation.wireless_environment.constants import MAP_SIZE

DEVICE_IMG = mpimg.imread(os.path.join(BASE_DIR, "img", "device.png"))
AP_IMG = mpimg.imread(os.path.join(BASE_DIR, "img", "AP.png"))


def tint_image(img, color):
    """Apply RGBA color to a grayscale image."""
    img = 1 - img
    if img.ndim == 3:  # if already RGB (e.g. loaded PNG with 3 channels)
        img = img[..., 0]  # take one channel
    img = img / img.max()  # normalize grayscale to [0, 1]

    # Create tinted RGBA
    tinted = np.zeros((*img.shape, 4))
    for i in range(3):  # R, G, B
        tinted[..., i] = img * color[i]
    tinted[..., 3] = img * color[3]  # Alpha
    return tinted


# --- Custom legend handler to show PNG icons ---
class ImageHandler(HandlerBase):
    def __init__(self, image: np.ndarray):
        self.image = image
        super(ImageHandler, self).__init__()

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # Scale image so height matches text height
        img_height = fontsize
        img_width = img_height * self.image.shape[1] / self.image.shape[0]

        # Center vertically relative to text line
        bb = Bbox.from_bounds(
            xdescent + (width - img_width) / 2,
            ydescent + (height - img_height) / 2,
            img_width,
            img_height,
        )
        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image)

        self.update_prop(image, orig_handle, legend)
        return [image]


def plot_positions(ax: Axes, positions: List[Dict], colors: Colormap) -> None:
    """
    Plot APs, IoT devices and obstacles positions

    Parameters
    ----------
    ax : Axes
        The ax the plot in
    positions : Dict
        The positions of APs, IoT devices and obstacles
    colors : Colormap
        The colormap for each cluster
    """
    ax.set_title("Access Points and IoT Devices Positions")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-MAP_SIZE[0] / 2, MAP_SIZE[0] / 2)
    ax.set_ylim(-MAP_SIZE[1] / 2, MAP_SIZE[1] / 2)

    legend_handles = []
    legend_labels = []
    handler_map = {}

    for idx, cluster in enumerate(positions):
        cid = cluster["ID"]
        ap_x, ap_y = cluster["AP"]
        dev_positions = np.array(cluster["devices"])
        dids = [f"{did + 1}" for did in range(len(dev_positions))]
        obstacle_positions = cluster["obstacles"]
        ap_img = tint_image(AP_IMG, color=colors[idx])
        device_img = tint_image(DEVICE_IMG, color=colors[idx])

        # --- Plot AP ---
        ap_imgbox = OffsetImage(ap_img, zoom=0.3)
        ap_ab = AnnotationBbox(ap_imgbox, (ap_x, ap_y), frameon=False)
        ax.add_artist(ap_ab)

        # Add AP legend entry
        ap_proxy = Line2D([0], [0])  # invisible proxy
        legend_handles.append(ap_proxy)
        legend_labels.append(f"Cluster {cid+1} Access Point")
        handler_map[ap_proxy] = ImageHandler(ap_img)

        # --- Plot devices ---
        for xi, yi in dev_positions:
            imagebox = OffsetImage(device_img, zoom=0.2)
            ab = AnnotationBbox(imagebox, (xi, yi), frameon=False)
            ax.add_artist(ab)

        for (x, y), did in zip(dev_positions, dids):
            ax.text(x, y, s=did, fontsize=12, ha="center", va="center")

        # Add Devices legend entry
        device_proxy = Line2D([0], [0])  # invisible proxy
        legend_handles.append(device_proxy)
        legend_labels.append(f"Cluster {cid+1} Devices")
        handler_map[device_proxy] = ImageHandler(device_img)

        # --- Plot obstacles ---
        for pos in obstacle_positions:
            start_point, end_point = pos
            x_coords = [start_point[0], end_point[0]]
            y_coords = [start_point[1], end_point[1]]
            ax.plot(x_coords, y_coords, color="k", linewidth=4)

    # Add obstacles legend entry (use Line2D directly)
    legend_handles.append(Line2D([0], [0], color="k", linewidth=3))
    legend_labels.append("Obstacles")

    # --- Build legend ---
    ax.legend(
        legend_handles,
        legend_labels,
        handler_map=handler_map,
        # bbox_to_anchor=(1.00, 1),  # shift legend outside to the right
        loc="upper left",
        fontsize=20,
    )
    ax.grid(True, linestyle="--", alpha=0.5)


if __name__ == "__main__":
    import json
    from multi_agent_power_allocation.wireless_environment.utils import rotate_points

    scenario = "scenario_4"
    data_path = os.path.join(BASE_DIR, "data", scenario)
    cluster_folders = sorted(
        [
            name
            for name in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, name))
        ]
    )
    print(cluster_folders)

    positions: List[Dict] = []
    for cluster_id, cluster_data_folder in enumerate(cluster_folders):
        with open(
            os.path.join(data_path, cluster_data_folder, "positions.json"), "rt"
        ) as file:
            positions.append(json.load(file))
            positions[-1].update({"ID": cluster_id})

    ax = plt.subplot()
    ax.set_aspect("equal", adjustable="box")

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i) for i in range(len(positions))]  # first two: blue and orange
    print(colors)
    # plot_positions(ax, positions, colors=colors)
    # plt.show()

    for position in positions:
        position["obstacles"][0] = rotate_points(
            position["obstacles"][0], -90, position["AP"]
        )

    for position in positions:
        position["obstacles"][0] = rotate_points(
            position["obstacles"][0], 90, position["AP"]
        )
    plot_positions(ax, positions, colors=colors)
    plt.show()
