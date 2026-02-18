
import os
from nd2reader import Nd2
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import tkinter as tk
from tkinter import filedialog, messagebox
from scipy.interpolate import splprep, splev


def normalize_image(
        img_array: np.ndarray
):
    """Normalizes an image using MinMaxScaling.

    Parameters
    ----------
        img_array : np.ndarray 
            An image array of shape (height, width)

    Returns
    -------
        norm_image : np.ndarray(dtype=np.uint8)
            A normalized image with integer pixel values. Returns all zeros if 
            pixels have equivalent values.
    """
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max == img_min:
        return np.zeros_like(img_array, dtype=np.uint8)
    norm_img = (img_array - img_min) / (img_max - img_min) * 255
    return norm_img.astype(np.uint8)


def grayscale_to_green_rgb(
        gray_img: np.ndarray
):
    """Generates a green image of cells given a grayscale image

    Parameters
    ----------
        gray_img : np.ndarray((h,w), dtype=uint8)
            A grayscale image with dimensions (h, w)

    Returns
    -------
        rgb_img : np.ndarray((h,w, 3), dtype=uint8)
            RGB image with green channel (h,w,1) values
    """
    # Assumes gray_img is a 2D uint8 NumPy array
    h, w = gray_img.shape
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_img[..., 1] = gray_img  # Set green channel
    return rgb_img


def extract_nd2_frames_to_png(
        nd2_filepath: Path,
        output_folder: Path,
        channel_name: str = "Green"
):
    """Extracts each frame of a .ND2 file into colorized and grayscale .PNG's.

    Parameters
    ----------
        nd2_filepath : Path 
            Path of .ND2 file to be analyzed
        output_folder : Path
            Path to save output .PNG files to
        channel_name : str, optional 
            Name of image channel to extract intensities. Defaults to "Green".

    Returns
    -------
        int: Total number of frames extracted from .ND2 file

    Raises
    ------
        RuntimeError 
            If no channel metadata is found in .ND2 file
        ValueError
            If the provided channel_name isn't found in .ND2 file
        ValueError
            If .ND2 frame contains unexpected dimensions
    """

    # Make output path absolute, check if it exists
    output_folder = Path(output_folder).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    # Open the Nd2 stack as images
    with Nd2(str(nd2_filepath)) as images:

        # ---------- ERROR_CHECK ----------
        # Check if channel contains metadata, if none is found, stop
        if not hasattr(images, "channels") or not images.channels:
            raise RuntimeError("No channel metadata found in ND2 file.")

        # Print available channels
        print(f"Available channels: {images.channels}")

        # ---------- VALUE_CHECK ----------
        # If the received channel_name wasn't found, stop
        if channel_name not in images.channels:
            raise ValueError(
                f"Channel '{channel_name}' not found. Available: {images.channels}")

        # Get the numerical index of the desired channel to be analyzed
        channel_index = images.channels.index(channel_name)

        # Enumerate over all frames
        for i, frame in enumerate(images):

            if frame.ndim == 3:
                # If frame has has 3 dimensions (channels, height, width), recast
                # a 2D array with dimensions (height, width)
                channel_frame = frame[channel_index]
            elif frame.ndim == 2:
                # If frame has single color channel, assume it's the correct one
                channel_frame = frame
            else:
                # ---------- VALUE_CHECK ----------
                # If frame has an unexpected number of dimensions, stop
                raise ValueError(f"Unexpected frame shape: {frame.shape}")

            # Print frame parameters
            print(f"Frame {i}:")
            print(f"    Minumum pixel intensity: {channel_frame.min()},")
            print(f"    Maximum pixel intensity: {channel_frame.max()},")
            print(f"    Frame Data Shape: {channel_frame.shape}")

            # Normalize frame with MinMaxScaling
            norm_frame = normalize_image(channel_frame)
            # Generate a color pixel data of frame
            green_rgb_frame = grayscale_to_green_rgb(channel_frame)
            # Generate color image from color pixel data
            img = Image.fromarray(green_rgb_frame)
            img.save(output_folder / f"green_frame_{i:03d}.png")
            # Generate grayscale image from grayscale pixel data
            img = Image.fromarray(norm_frame)
            img.save(output_folder / f"norm_frame_{i:03d}.png")

            print(output_folder / f"frame_{i:03d}.png")

    # Return total frame count
    return i + 1


def crop_to_cell(
        batch: str,
        single_ROI: bool,
        screen_width: int,
        screen_height: int,
        debug: bool = False
):
    """Takes a path of .ND2 whole frame images, allows user to select ROI for
    further segmentation and analysis from these images. Users can specify a
    single ROI or specify ROI per frame. Saves all cropped outputs in a nested
    output folder for further analysis.

    Parameters
    ----------
    batch : str
        Name of output directory provided at beginning of routine
    single_ROI : bool
        Select whether a single ROI is used for all frames or user would like
        to specify ROI per frame
    screen_width : int (UNUSED)
        Detected width of user's screen
    screen_height : int (UNUSED)
        Detected height of user's screen
    debug : bool, optional
        Optional debug flag for monitoring contour detection, by default False
    """

    # Path to .ND2 frame .PNGS, temp_out is created in same directory as code
    # by default
    image_dir = Path("temp_out") / batch

    # Initialize ROI variable, needed for a check later
    roi = None

    # Sort filenames to keep frame order consistent in final timelapse plot
    filenames = sorted(f for f in os.listdir(image_dir)
                       if not f.startswith("green_frame"))

    # Iterate through the colorized .ND2 frames
    for i, filename in enumerate(filenames):

        # Set file path
        file_path = os.path.join(image_dir, filename)

        # Read frame as grayscale image
        frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        # Equalize distribution of pixel intensities
        enhanced = cv2.equalizeHist(frame)

        # Check for one of two cases:
        #   - If this is the first frame and the user has elected to use a single
        #   ROI across all frames, then roi is None on first pass, but contains
        #   data and returns True on subsequent iterations.
        #   - If the user has elected to draw the ROI on each frame, then
        #   single_ROI will be False, returning True here
        if roi is None or not single_ROI:

            # Generate a window object for prompting user to select the ROI
            cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)

            # Resize the window object for better display of cropped image
            win_h, win_w = enhanced.shape
            cv2.resizeWindow("Select ROI", win_w, win_h)
            cv2.imshow("Select ROI", enhanced)
            cv2.waitKey(1)
            cv2.moveWindow("Select ROI", 0, 0)
            cv2.waitKey(1)

            roi = cv2.selectROI(
                "Select ROI",
                enhanced,
                fromCenter=False,
                showCrosshair=True
            )
            cv2.destroyAllWindows()

        # Return height, width, and (x, y) of lower left corner of ROI
        x, y, w, h = roi
        # Reduce frame to the ROI selected
        cropped = frame[y:y+h, x:x+w]

        # If debug=True, display the cropped visual to verify region selection
        if debug:
            cropped_visual = enhanced[y:y+h, x:x+w]
            cv2.imshow(f"Cropped Region - Frame {i}", cropped_visual)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Generate a nested output folder to contain cropped images
        output = Path(image_dir) / "cropped"
        output.mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(output, f"cropped_{filename}")
        cv2.imwrite(out_file, cropped)

    return


def process_crops(
        batch: str
):
    """Processes a folder of cropped ROI .PNG files by segmenting and 
    extracting the largest detected contour, which is assumed to be the cell
    of interest

    Parameters
    ----------
    batch : str
        Name of output directory containing .ND2 frame images and a 
        subdirectory of cropped frames centered on ROI.
    """

    def segment_and_extract_cell_contours(
            image: np.ndarray
    ):
        """Internal function used to segment and extract cellular contours from
        cropped images.

        Parameters
        ----------
            image : ndarray
                Numpy array of pixel values for an image 

        Returns
        -------
            largest : ndarray(N, 1, 2)
                Numpy array containing
                    - number of points on perimeter
                    - Spare dimensions for openCV compatibility?
                    - (x, y) coordinates of each perimeter point
        """

        # Denoise the image with a Gaussian blur
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        # Threshold using Otsu Thresholding to seperate cell from background
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use OpenCV to find all contours in the thresholded image
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If no contours were detected, return None
        if not contours:
            return None

        # Select largest contour based on area
        largest = max(contours, key=cv2.contourArea)

        # Return largest contour
        return largest

    # Set path to folder of cropped images, as these will be used to segment
    # and extract cellular contours
    cropped_folder = Path('temp_out') / batch / 'cropped'
    # Initialize empty results list
    results = []

    # Iterate through the ROI cropped images
    for crop in sorted(os.listdir(cropped_folder)):

        # If a file does not end with .PNG, skip it (unanticipated file)
        if not crop.endswith(".png"):
            continue

        # Set path to the file
        filepath = cropped_folder / crop

        # Read the cropped image file as grayscale openCv image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # Segment and extract the largest cell contour
        contour = segment_and_extract_cell_contours(img)

        if contour is not None:
            # If a contour was found, squeeze out the extra OpenCV dimension
            boundary = contour.squeeze().tolist()

            if isinstance(boundary[0], int):
                # If the boundary is only single point for some reason, reformat
                boundary = [boundary]
        else:
            boundary = []

        # Append detected cell boundary to the results list
        results.append({
            "filename": crop,
            "boundary": boundary
        })

    output_json = cropped_folder / 'contour.json'

    # Save the output list of boundaries into a .JSON file
    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)

    return


def plot_jagged_contours(
        background_img_path: Path,
        contour_json_path: Path,
        output_path: Path,
        step: int = 2
):
    """Plots a timelapse of all detected cell contours without any applied 
    smoothing. Contours will appear jagged, as the plotted traces connect 
    subsequent boundary points with straight lines.

    Parameters
    ----------
    background_img_path : Path
        The first image in the time series of cropped images. Used as 
        background for generated figure
    contour_json_path : Path
        Path to .JSON file containing boundaries and coordinates of each frame
        for cell contour
    output_path : Path
        Path to save output image to
    step : int, optional
        Step size for moving through contours, by default 2
    """

    # Load the grayscale base image (background)
    base_img = cv2.imread(background_img_path, cv2.IMREAD_GRAYSCALE)
    # Load the grayscale image (but as color) for the purpose of plotting
    color_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)

    # Load contour data from .JSON
    with open(contour_json_path, 'r') as f:
        data = json.load(f)

    n = len(data)

    # Create a colormap (you can change "viridis", "plasma", etc.)
    colormap = cm.get_cmap('viridis', n)

    # Iterate through contours by step size
    for i in range(0, n, step):

        entry = data[i]
        boundary = entry["boundary"]
        if not boundary:
            continue  # skip empty boundaries

        contour = np.array(boundary).reshape((-1, 1, 2)).astype(np.int32)

        # Get color from colormap
        rgba = colormap(i)

        # Convert from float to 8-bit BGR
        color = tuple(int(255 * c) for c in rgba[:3])

        # Convert RGB to BGR
        color_bgr = color[::-1]

        # Draw contour
        cv2.drawContours(color_img, [contour], -1, color_bgr, thickness=1)

    # Convert BGR image to RGB for matplotlib display
    img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(1, 20, wspace=0.0)
    ax_cbar = fig.add_subplot(gs[:, 0])       # Narrow left axis for colorbar
    ax_img = fig.add_subplot(gs[:, 1:])       # Main axis for image

    ax_img.imshow(img_rgb)
    ax_img.axis('off')
    ax_img.set_title("Protrusion Dynamics Overlay")
    # Create ScalarMappable for colorbar with same colormap and normalization
    norm = colors.Normalize(vmin=0, vmax=n-1)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])

    # Create vertical colorbar on the left axis
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.tick_left()

    # Label colorbar and invert y-axis for "Start" at bottom, "End" at top
    cbar.set_label("Time (min)", rotation=90, fontsize=12, labelpad=-18)
    cbar.set_ticks([0, n-1])
    cbar.set_ticklabels(["0", "30"])

    plt.tight_layout()
    plt.savefig(output_path)

    return


def plot_smooth_contours(
        background_img_path: Path,
        contour_json_path: Path,
        output_path: Path,
        scale_factor: int = 5,
        use_spline: bool = True,
        smoothing_factor: float = 1.0,
        points_per_contour: int = 200,
        step: int = 2
):
    """Plots a timelapse of all detected cell contours using B-splines to 
    smooth cell contours.

    Parameters
    ----------
    background_img_path : Path
        The first image in the time series of cropped images. Used as 
        background for generated figure
    contour_json_path : Path
        Path to .JSON file containing boundaries and coordinates of each frame
        for cell contour
    output_path : Path
        Path to save output image to
    scale_factor : int, optional
        Multiplier used to upscale image and coordinates, by default 5.
        Provides a higher resolution ensuring that smoothed curves do not 
        appear pixelated
    use_spline : bool, optional
        Flag for optional B-splining, by default True
    smoothing_factor : float, optional
        Parameter for b-spline controlling level of smoothness, by default 1.0
    points_per_contour : int, optional
        Number of smoothed points to place on contour, by default 200
    step : int, optional
        Step size for moving through contours, by default 2
    """

    # Load the grayscale base image (background)
    base_img = cv2.imread(background_img_path, cv2.IMREAD_GRAYSCALE)
    # Get image dimensions
    img_h, img_w = base_img.shape

    # Upscale image
    hr_h, hr_w = img_h * scale_factor, img_w * scale_factor
    hr_img = cv2.resize(base_img, (hr_w, hr_h),
                        interpolation=cv2.INTER_NEAREST)
    hr_img_color = cv2.cvtColor(hr_img, cv2.COLOR_GRAY2BGR)

    # Load contour data from .JSON
    with open(contour_json_path, 'r') as f:
        data = json.load(f)

    n = len(data)

    # Create a colormap (you can change "viridis", "plasma", etc.)
    colormap = cm.get_cmap('viridis', n)

    # Iterate through contours by step size
    for i in range(0, n, step):

        entry = data[i]
        boundary = entry["boundary"]
        if not boundary:
            continue    # skip empty boundaries

        contour = np.array(boundary, dtype=float)

        # Upscale
        x = contour[:, 0] * scale_factor
        y = contour[:, 1] * scale_factor

        # Remove large jumps or irregularities between consecutive points
        dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        median_dist = np.median(dists)
        jump_thresh = median_dist * 5  # adjust factor if needed
        good_idx = np.where(dists <= jump_thresh)[0]
        # Always include the last point
        good_idx = np.append(good_idx, len(x)-1)
        x, y = x[good_idx], y[good_idx]

        # Apply B-spline smoothing if requested and enough points
        if use_spline and len(x) > 3:
            try:
                # Closed spline for contours
                tck, u = splprep([x, y], s=smoothing_factor, per=True)
                u_new = np.linspace(0, 1, points_per_contour)
                x_smooth, y_smooth = splev(u_new, tck)
            except Exception:
                x_smooth, y_smooth = x, y
        else:
            x_smooth, y_smooth = x, y

        # Clamp to image bounds
        x_smooth = np.clip(x_smooth, 0, hr_w-1)
        y_smooth = np.clip(y_smooth, 0, hr_h-1)

        # Stack points for polylines
        polyline = np.stack([x_smooth, y_smooth], axis=-
                            1).astype(np.int32).reshape(-1, 1, 2)

        # Get color from colormap
        rgba = colormap(i)

        # Convert from float to 8-bit BGR
        color = tuple(int(255 * c) for c in rgba[:3])

        # Convert RGB to BGR
        color_bgr = color[::-1]

        # Draw polyline
        cv2.polylines(hr_img_color, [polyline],
                      isClosed=True, color=color_bgr, thickness=2)

    # Display and save
    img_rgb = cv2.cvtColor(hr_img_color, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(6, 6))
    gs = fig.add_gridspec(1, 20, wspace=0.0)
    ax_cbar = fig.add_subplot(gs[:, 0])
    ax_img = fig.add_subplot(gs[:, 1:])
    ax_img.imshow(img_rgb)
    ax_img.axis('off')
    ax_img.set_title("Protrusion Dynamics Overlay (Safe Smooth)")

    norm = colors.Normalize(vmin=0, vmax=n-1)
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar, orientation='vertical')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.tick_left()
    cbar.set_label("Time (min)", rotation=90, fontsize=12, labelpad=-18)
    cbar.set_ticks([0, n-1])
    cbar.set_ticklabels(["0", "30"])

    plt.tight_layout()
    plt.savefig(output_path)

    return


if __name__ == "__main__":

    print("Beginning routine")

    this_dir = Path(__file__).parent
    os.chdir(this_dir)

    root = tk.Tk()
    root.withdraw()

    screen_width = root.winfo_screenmmwidth()
    screen_height = root.winfo_screenmmheight()

    file_path = filedialog.askopenfilename(parent=root)
    print(f"Path is {file_path}")
    name = input("Provide a meaningful prefix for the output images: ")

    out_temp = Path("temp_out")
    print(out_temp)
    frames = extract_nd2_frames_to_png(file_path, out_temp / name)

    print(f"Image extraction complete, {frames} frames extracted")

    # Set Single_Roi to true to only draw one ROI, set to draw per frame
    use_single_ROI = messagebox.askyesno(
        "ROI Selection Mode",
        "Would you like to use a single ROI for all frames in the file?"
    )
    root.destroy()

    if use_single_ROI is None:
        raise RuntimeError(
            "ROI Selection dialog was closed without a selection."
        )

    crop_to_cell(
        batch=name,
        single_ROI=use_single_ROI,
        screen_height=screen_height,
        screen_width=screen_width
    )

    process_crops(batch=name)

    jagged_image_name = str(name) + "_overlaid_jagged_contours.png"
    jagged_image_name = Path(jagged_image_name)

    plot_jagged_contours(
        background_img_path=out_temp / name /
        "cropped/cropped_norm_frame_000.png",  # the first frame
        contour_json_path=out_temp / name / "cropped/contour.json",
        output_path=out_temp / jagged_image_name
    )

    smooth_image_name = str(name) + "_overlaid_smooth_contours.png"
    smooth_image_name = Path(smooth_image_name)

    plot_smooth_contours(
        background_img_path=out_temp / name /
        "cropped/cropped_norm_frame_000.png",  # the first frame
        contour_json_path=out_temp / name / "cropped/contour.json",
        output_path=out_temp / smooth_image_name,
        smoothing_factor=1,
        points_per_contour=200
    )

    print("This script has completed!")
