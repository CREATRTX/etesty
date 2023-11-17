from nuscenes.panoptic.panoptic_utils import get_frame_panoptic_instances, get_panoptic_instances_stats
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_io import load_bin_file


def truncate_class_name(class_name: str) -> str:
    """
    Truncate a given class name according to a pre-defined map.
    :param class_name: The long form (i.e. original form) of the class name.
    :return: The truncated form of the class name.
    """
from nuscenes.panoptic.panoptic_utils import get_frame_panoptic_instances, get_panoptic_instances_stats
from nuscenes.utils.color_map import get_colormap
from nuscenes.utils.data_io import load_bin_file


def truncate_class_name(class_name: str) -> str:
    """
    Truncate a given class name according to a pre-defined map.
    :param class_name: The long form (i.e. original form) of the class name.
    :return: The truncated form of the class name.
    """

    string_mapper = {
        "noise": 'noise',
        "human.pedestrian.adult": 'adult',
        "human.pedestrian.child": 'child',
        "human.pedestrian.wheelchair": 'wheelchair',
        "human.pedestrian.stroller": 'stroller',
        "human.pedestrian.personal_mobility": 'p.mobility',
        "human.pedestrian.police_officer": 'police',
        "human.pedestrian.construction_worker": 'worker',
        "animal": 'animal',
        "vehicle.car": 'car',
        "vehicle.motorcycle": 'motorcycle',
        "vehicle.bicycle": 'bicycle',
        "vehicle.bus.bendy": 'bus.bendy',
        "vehicle.bus.rigid": 'bus.rigid',
        "vehicle.truck": 'truck',
        "vehicle.construction": 'constr. veh',
        "vehicle.emergency.ambulance": 'ambulance',
        "vehicle.emergency.police": 'police car',
        "vehicle.trailer": 'trailer',
        "movable_object.barrier": 'barrier',
        "movable_object.trafficcone": 'trafficcone',
        "movable_object.pushable_pullable": 'push/pullable',
        "movable_object.debris": 'debris',
        "static_object.bicycle_rack": 'bicycle racks',
        "flat.driveable_surface": 'driveable',
        "flat.sidewalk": 'sidewalk',
        "flat.terrain": 'terrain',
        "flat.other": 'flat.other',
        "static.manmade": 'manmade',
        "static.vegetation": 'vegetation',
        "static.other": 'static.other',
        "vehicle.ego": "ego"
    }

    return string_mapper[class_name]


def render_histogram(nusc: NuScenes,
                     sort_by: str = 'count_desc',
                     verbose: bool = True,
                     font_size: int = 20,
                     save_as_img_name: str = None) -> None:
    """
    Render two histograms for the given nuScenes split. The top histogram depicts the number of scan-wise instances
    for each class, while the bottom histogram depicts the number of points for each class.
    :param nusc: A nuScenes object.
    :param sort_by: How to sort the classes to display in the plot (note that the x-axis, where the class names will be
        displayed on, is shared by the two histograms):
        - count_desc: Sort the classes by the number of points belonging to each class, in descending order.
        - count_asc: Sort the classes by the number of points belonging to each class, in ascending order.
        - name: Sort the classes by alphabetical order.
        - index: Sort the classes by their indices.
    :param verbose: Whether to display the plot in a window after rendering.
    :param font_size: Size of the font to use for the plot.
    :param save_as_img_name: Path (including image name and extension) to save the plot as.
    string_mapper = {
        "noise": 'noise',
        "human.pedestrian.adult": 'adult',
        "human.pedestrian.child": 'child',
        "human.pedestrian.wheelchair": 'wheelchair',
        "human.pedestrian.stroller": 'stroller',
        "human.pedestrian.personal_mobility": 'p.mobility',
        "human.pedestrian.police_officer": 'police',
        "human.pedestrian.construction_worker": 'worker',
        "animal": 'animal',
        "vehicle.car": 'car',
        "vehicle.motorcycle": 'motorcycle',
        "vehicle.bicycle": 'bicycle',
        "vehicle.bus.bendy": 'bus.bendy',
        "vehicle.bus.rigid": 'bus.rigid',
        "vehicle.truck": 'truck',
        "vehicle.construction": 'constr. veh',
        "vehicle.emergency.ambulance": 'ambulance',
        "vehicle.emergency.police": 'police car',
        "vehicle.trailer": 'trailer',
        "movable_object.barrier": 'barrier',
        "movable_object.trafficcone": 'trafficcone',
        "movable_object.pushable_pullable": 'push/pullable',
        "movable_object.debris": 'debris',
        "static_object.bicycle_rack": 'bicycle racks',
        "flat.driveable_surface": 'driveable',
        "flat.sidewalk": 'sidewalk',
        "flat.terrain": 'terrain',
        "flat.other": 'flat.other',
        "static.manmade": 'manmade',
        "static.vegetation": 'vegetation',
        "static.other": 'static.other',
        "vehicle.ego": "ego"
    }

    return string_mapper[class_name]


def render_histogram(nusc: NuScenes,
                     sort_by: str = 'count_desc',
                     verbose: bool = True,
                     font_size: int = 20,
                     save_as_img_name: str = None) -> None:
    """
    Render two histograms for the given nuScenes split. The top histogram depicts the number of scan-wise instances
    for each class, while the bottom histogram depicts the number of points for each class.
    :param nusc: A nuScenes object.
    :param sort_by: How to sort the classes to display in the plot (note that the x-axis, where the class names will be
        displayed on, is shared by the two histograms):
        - count_desc: Sort the classes by the number of points belonging to each class, in descending order.
        - count_asc: Sort the classes by the number of points belonging to each class, in ascending order.
        - name: Sort the classes by alphabetical order.
        - index: Sort the classes by their indices.
    :param verbose: Whether to display the plot in a window after rendering.
    :param font_size: Size of the font to use for the plot.
    :param save_as_img_name: Path (including image name and extension) to save the plot as.
    """

    # Get the statistics for the given nuScenes split.
    lidarseg_num_points_per_class = get_lidarseg_num_points_per_class(nusc, sort_by=sort_by)
    panoptic_num_instances_per_class = get_panoptic_num_instances_per_class(nusc, sort_by=sort_by)

    # Align the two dictionaries by adding entries for the stuff classes to panoptic_num_instances_per_class; the
    # instance count for each of these stuff classes is 0.
    panoptic_num_instances_per_class_tmp = dict()
    for class_name in lidarseg_num_points_per_class.keys():
        num_instances_for_class = panoptic_num_instances_per_class.get(class_name, 0)
        panoptic_num_instances_per_class_tmp[class_name] = num_instances_for_class
    panoptic_num_instances_per_class = panoptic_num_instances_per_class_tmp

    # Define some settings for each histogram.
    histograms_config = dict({
        'panoptic': {
            'y_values': list(panoptic_num_instances_per_class.values()),
            'y_label': 'No. of instances',
            'y_scale': 'log'
        },
        'lidarseg': {
            'y_values': list(lidarseg_num_points_per_class.values()),
            'y_label': 'No. of lidar points',
            'y_scale': 'log'
        }
    })

    # Ensure the same set of class names are used for all histograms.
    assert lidarseg_num_points_per_class.keys() == panoptic_num_instances_per_class.keys(), \
        'Error: There are {} classes for lidarseg, but {} classes for panoptic.'.format(
            len(lidarseg_num_points_per_class.keys()), len(panoptic_num_instances_per_class.keys()))
    class_names = list(lidarseg_num_points_per_class.keys())

    # Create an array with the colors to use.
    cmap = get_colormap()
    colors = ['#%02x%02x%02x' % tuple(cmap[cn]) for cn in class_names]  # Convert from RGB to hex.

    # Make the class names shorter so that they do not take up much space in the plot.
    class_names = [truncate_class_name(cn) for cn in class_names]

    # Start a plot.
    fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(16, 9))
    for ax in axes:
        ax.margins(x=0.005)  # Add some padding to the left and right limits of the x-axis for aesthetics.
        ax.set_axisbelow(True)  # Ensure that axis ticks and gridlines will be below all other ploy elements.
        ax.yaxis.grid(color='white', linewidth=2)  # Show horizontal gridlines.
        ax.set_facecolor('#eaeaf2')  # Set background of plot.
        ax.spines['top'].set_visible(False)  # Remove top border of plot.
        ax.spines['right'].set_visible(False)  # Remove right border of plot.
        ax.spines['bottom'].set_visible(False)  # Remove bottom border of plot.
        ax.spines['left'].set_visible(False)  # Remove left border of plot.

    # Plot the histograms.
    for i, (histogram, config) in enumerate(histograms_config.items()):
        axes[i].bar(class_names, config['y_values'], color=colors)
        assert len(class_names) == len(axes[i].get_xticks()), \
            'There are {} classes, but {} are shown on the x-axis'.format(len(class_names), len(axes[i].get_xticks()))

        # Format the x-axis.
        axes[i].set_xticklabels(class_names, rotation=45, horizontalalignment='right',
                                fontweight='light', fontsize=font_size)

        # Shift the class names on the x-axis slightly to the right for aesthetics.
        trans = mtrans.Affine2D().translate(10, 0)
        for t in axes[i].get_xticklabels():
            t.set_transform(t.get_transform() + trans)

        # Format the y-axis.
        axes[i].set_ylabel(config['y_label'], fontsize=font_size)
        axes[i].set_yticklabels(config['y_values'], size=font_size)
        axes[i].set_yscale(config['y_scale'])

        if config['y_scale'] == 'linear':
            axes[i].yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'))

    class BitMap:

        def __init__(self, dataroot: str, map_name: str, layer_name: str):
            """
            This class is used to render bitmap map layers. Currently these are:
            - semantic_prior: The semantic prior (driveable surface and sidewalks) mask from nuScenes 1.0.
            - basemap: The HD lidar basemap used for localization and as general context.

            :param dataroot: Path of the nuScenes dataset.
            :param map_name: Which map out of `singapore-onenorth`, `singepore-hollandvillage`, `singapore-queenstown` and
                'boston-seaport'.
            :param layer_name: The type of bitmap map, `semanitc_prior` or `basemap.
            """
            self.dataroot = dataroot
            self.map_name = map_name
            self.layer_name = layer_name

            self.image = self.load_bitmap()

        def load_bitmap(self) -> np.ndarray:
            """
            Load the specified bitmap.
            """
            # Load bitmap.
            if self.layer_name == 'basemap':
                map_path = os.path.join(self.dataroot, 'maps', 'basemap', self.map_name + '.png')
            elif self.layer_name == 'semantic_prior':
                map_hashes = {
                    'singapore-onenorth': '53992ee3023e5494b90c316c183be829',
                    'singapore-hollandvillage': '37819e65e09e5547b8a3ceaefba56bb2',
                    'singapore-queenstown': '93406b464a165eaba6d9de76ca09f5da',
                    'boston-seaport': '36092f0b03a857c6a3403e25b4b7aab3'
                }
                map_hash = map_hashes[self.map_name]
                map_path = os.path.join(self.dataroot, 'maps', map_hash + '.png')
            else:
                raise Exception('Error: Invalid bitmap layer: %s' % self.layer_name)

            # Convert to numpy.
            if os.path.exists(map_path):
                image = np.array(Image.open(map_path))
            else:
                raise Exception('Error: Cannot find %s %s! Please make sure that the map is correctly installed.'
                                % (self.layer_name, map_path))

            # Invert semantic prior colors.
            if self.layer_name == 'semantic_prior':
                image = image.max() - image

            return image

        def render(self, canvas_edge: Tuple[float, float], ax: Axis = None):
            """
            Render the bitmap.
            Note: Regardless of the image dimensions, the image will be rendered to occupy the entire map.
            :param canvas_edge: The dimension of the current map in meters (width, height).
            :param ax: Optional axis to render to.
            """
            if ax is None:
                ax = plt.subplot()
            x, y = canvas_edge
            if len(self.image.shape) == 2:
                ax.imshow(self.image, extent=[0, x, 0, y], cmap='gray')
            else:
                ax.imshow(self.image, extent=[0, x, 0, y])
