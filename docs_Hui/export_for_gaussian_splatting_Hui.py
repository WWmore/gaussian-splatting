# Exports undistorted photos, internal/external camera orientations, tie points in a simple COLMAP format.
# Exported COLMAP project can be used as input for Gaussian Splatting. See how to use it - https://github.com/PolarNick239/gaussian-splatting-Windows#requirements
#
# Usecase: photo alignment was done in Metashape, export for Gaussian Splatting is required.
#
# Default parameters for GUI can be changed in ExportSceneParams.__init__
#
# Options:
# -- Enforce zero cx, cy -- output camera calibrations will have zero cx and cy.
#        May result in information loss during export (large cropping).
#        Should be checked until Gaussian Splatting software considers this parameters.
# -- Use localframe -- shift coordinates origin to the center of the bounding box, use localframe rotation at this point
# -- Image quality -- quality of the output undistorted images (jpeg only), min 0, max 100
#
# Exported files structure (for all chunks and all frames):
# <chosen_folder>
# |---<chunk_0_folder>
# |   |---<frame_0_folder>
# |   |   |---images
# |   |   |   |---<image_0>
# |   |   |   |---<image_1>
# |   |   |   |---...
# |   |   |---sparse
# |   |       |---0
# |   |           |---cameras.bin
# |   |           |---images.bin
# |   |           |---points3D.bin
# |   |---<frame_1_folder>
# |   |---...
# |---<chunk_1_folder>
# |---...
#

# """
# Usage:
# 1. Align photos
# 1.1.   Workflow -> Add Photos...
# 1.2.   Workflow -> Align Photos...
# 1.3. Hui: chunk1 / import / impoart points / choose .las file --> wait --> lidar data imported successfully --> click `Point Cloud` to show
# 1.4. Hui: change all the `tie_points` in this file into `tie_points` which is the imported lidar file name

# 2. Run script
# 2.1.   Tools -> Run script... then choose this script, without arguments
# 2.2.   Scripts -> Export Colmap project (for Gaussian Splatting)
# 2.3.   Click Export
# 2.4.   Choose destination folder (for several chunks/frames additional subfolders will be added)
# Hui: two new folders `images` and `sparse` will be produced; move them into data_Dji_L2_lidar
# Hui: details in the popup windown in Metashape when clicking `Script` as seen above.

# Hui: 3. Train the exported
# Hui: 3.1 open anaconda prompt
# Hui: 3.2 conda activate gaussian_splatting
# Hui: 3.3 cd C:\Users\NAME\gaussian-splatting
# Hui: 3.4 python train.py -s data_Dji_L2_lidar
# Hui: or
# Hui: 3.4 python train.py -s data_Dji_L2_lidar -r 1
# where -r 1 argument - thanks to it, camera images will be used without downscale. Huinote: careful to use, since it's easily out of memory
# Hui: 3.5 rename folder in output to out_Dji_L2_lidar
# """

# Hui note: three functions relate to frame.tie_points: `get_filtered_track_structure`, `save_points`, `export_for_gaussian_splatting`
# Hui add function: `replace_tie_points_by_lidar`, `read_lidar`, `get_replaced_tie_points`

# cmd run as administor; "%programfiles%\Agisoft\Metashape Pro\python\python.exe" -m pip install numpy open3d laspy

import os
import shutil
import struct
import math
from PySide2 import QtGui, QtCore, QtWidgets
import open3d as o3d
import numpy as np
import laspy

# Checking compatibility
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


f32 = lambda x: bytes(struct.pack("f", x))
d64 = lambda x: bytes(struct.pack("d", x))
u8  = lambda x: x.to_bytes(1, "little", signed=(x < 0))
u32 = lambda x: x.to_bytes(4, "little", signed=(x < 0))
u64 = lambda x: x.to_bytes(8, "little", signed=(x < 0))
bstr = lambda x: bytes((x + "\0"), "utf-8")

def matrix_to_quat(m):
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if (tr > 0):
        s = 2 * math.sqrt(tr + 1)
        return Metashape.Vector([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    if (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = 2 * math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2])
        return Metashape.Vector([0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if (m[1, 1] > m[2, 2]):
        s = 2 * math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2])
        return Metashape.Vector([(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    else:
        s = 2 * math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1])
        return Metashape.Vector([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s])


def get_camera_name(cam):
    name = cam.label
    ext = os.path.splitext(name)
    if (len(ext[1]) == 0):
        name = ext[0] + os.path.splitext(cam.photo.path)[1]
    return name

def clean_dir(folder, confirm_deletion):
    if os.path.exists(folder):
        if confirm_deletion:
            ok = Metashape.app.getBool('Folder "' + folder + '" will be deleted.\nAre you sure you want to continue?')
            if not ok:
                return False
        shutil.rmtree(folder)
    os.mkdir(folder)
    return True

def build_dir_structure(folder, confirm_deletion):
    if not os.path.exists(folder):
        os.makedirs(folder)
    if not clean_dir(folder + "images/", confirm_deletion):
        return False
    if not clean_dir(folder + "sparse/", confirm_deletion):
        return False
    os.makedirs(folder + "sparse/0/")
    return True

def get_chunk_dirs(folder, params):
    doc = Metashape.app.document
    chunk_name_stats = {}
    chunk_names = {}

    initial_chunk_selected = doc.chunk.selected
    doc.chunk.selected = True

    for chunk in doc.chunks:
        if not params.all_chunks and not chunk.selected:
            continue
        label = chunk.label
        i = chunk_name_stats[label] = chunk_name_stats.get(label, 0)
        while True:
            name = folder + label + ("" if i == 0 else "_" + str(i)) + "/"
            i += 1
            if name not in chunk_names.values():
                chunk_names[chunk.key] = name
                chunk_name_stats[label] = i
                break

    doc.chunk.selected = initial_chunk_selected

    if not params.all_frames and len(chunk_names) == 1:
        return {chunk_key:folder for chunk_key in chunk_names}

    existed = [name for name in chunk_names.values() if os.path.exists(name)]
    if len(existed) > 0:
        ok = Metashape.app.getBool('These folders will be deleted:\n"' + '"\n"'.join(existed) + '"\nAre you sure you want to continue?')
        if not ok:
            return {}
    for name in existed:
        shutil.rmtree(name)
    return chunk_names

def compute_undistorted_calib(sensor, zero_cxy):
    border = 0 # in pixels, can be increased if black margins are on the undistorted images

    if sensor.type != Metashape.Sensor.Type.Frame:
        return Metashape.Calibration()

    calib_initial = sensor.calibration
    w = calib_initial.width
    h = calib_initial.height

    calib = Metashape.Calibration()
    calib.f = calib_initial.f
    calib.width = w
    calib.height = h

    left = -float("inf")
    right = float("inf")
    top = -float("inf")
    bottom = float("inf")

    for i in range(h):
        pt = calib.project(calib_initial.unproject(Metashape.Vector([0.5, i + 0.5])))
        left = max(left, pt.x)
        pt = calib.project(calib_initial.unproject(Metashape.Vector([w - 0.5, i + 0.5])))
        right = min(right, pt.x)
    for i in range(w):
        pt = calib.project(calib_initial.unproject(Metashape.Vector([i + 0.5, 0.5])))
        top = max(top, pt.y)
        pt = calib.project(calib_initial.unproject(Metashape.Vector([i + 0.5, h - 0.5])))
        bottom = min(bottom, pt.y)

    left = math.ceil(left) + border
    right = math.floor(right) - border
    top = math.ceil(top) + border
    bottom = math.floor(bottom) - border

    if zero_cxy:
        new_w = min(2 * right - w, w - 2 * left)
        new_h = min(2 * bottom - h, h - 2 * top)
        new_w -= (new_w + w) % 2
        new_h -= (new_h + h) % 2
        left = (w - new_w) // 2
        right = (w + new_w) // 2
        top = (h - new_h) // 2
        bottom = (h + new_h) // 2

    calib.width = max(0, right - left)
    calib.height = max(0, bottom - top)
    calib.cx = -0.5 * (right + left - w)
    calib.cy = -0.5 * (top + bottom - h)

    return calib

def check_undistorted_calib(sensor, calib):
    border = 0 # in pixels, can be increased if black margins are on the undistorted images

    calib_initial = sensor.calibration
    w = calib.width
    h = calib.height

    left = float("inf")
    right = -float("inf")
    top = float("inf")
    bottom = -float("inf")

    for i in range(h):
        pt = calib_initial.project(calib.unproject(Metashape.Vector([0.5, i + 0.5])))
        left = min(left, pt.x)
        pt = calib_initial.project(calib.unproject(Metashape.Vector([w - 0.5, i + 0.5])))
        right = max(right, pt.x)
    for i in range(w):
        pt = calib_initial.project(calib.unproject(Metashape.Vector([i + 0.5, 0.5])))
        top = min(top, pt.y)
        pt = calib_initial.project(calib.unproject(Metashape.Vector([i + 0.5, h - 0.5])))
        bottom = max(bottom, pt.y)

    print(left, right, top, bottom)
    if (left < 0.5 or calib_initial.width - 0.5 < right or top < 0.5 or calib_initial.height - 0.5 < bottom):
        print("!!! Wrong undistorted calib")
    else:
        print("Ok:")

def get_coord_transform(frame, use_localframe):
    if not use_localframe:
        return frame.transform.matrix
    if not frame.region:
        print("Null region, using world crs instead of local")
        return frame.transform.matrix
    fr_to_gc  = frame.transform.matrix
    gc_to_loc = frame.crs.localframe(fr_to_gc.mulp(frame.region.center))
    fr_to_loc = gc_to_loc * fr_to_gc
    return (Metashape.Matrix.Translation(-fr_to_loc.mulp(frame.region.center)) * fr_to_loc)



def compute_undistorted_calibs(frame, zero_cxy):
    print("Calibrations:")
    calibs = {} # { sensor_key: ( sensor, undistorted calibration ) }
    for sensor in frame.sensors:
        calib = compute_undistorted_calib(sensor, zero_cxy)
        if (calib.width == 0 or calib.height == 0):
            continue
        calibs[sensor.key] = (sensor, calib)
        print(sensor.key, calib.f, calib.width, calib.height, calib.cx, calib.cy)
        #check_undistorted_calib(sensor, calib)

    return calibs

def get_calibs(camera, calibs):
    s_key = camera.sensor.key
    if s_key not in calibs:
        print("Camera " + camera.label + " (key = " + str(camera.key) + ") has cropped/unsupported sensor (key = " + str(s_key) + ")")
        return (None, None)
    return (calibs[s_key][0].calibration, calibs[s_key][1])


#-------------------------------------------------------------------------
def replace_tie_points_by_lidar(tie_points, lidar_points, k=1, is_array=False):
    """replace tie_points by lidar_points[ind];
    in Dji_L2_lidar data: lidar_points are more than tie_points
    """
    if is_array:
        "suppose: tie_points, lidar_points are both np.array"
        lidar_pcd = o3d.geometry.PointCloud()
        lidar_pcd.points = o3d.utility.Vector3dVector(lidar_points)
        lidar_pcd_tree = o3d.geometry.KDTreeFlann(lidar_pcd)
        for i, tie in enumerate(tie_points):
            _, lidar_ind, _ = lidar_pcd_tree.search_knn_vector_3d(tie, k)
            tie_points[i] = lidar_points[lidar_ind] ##TypeError: 'open3d.cpu.pybind.geometry.PointCloud' object is not subscriptable
        return tie_points
    else:
        "suppose: tie_points, lidar_points are both open3d"
        lidar_pcd_tree = o3d.geometry.KDTreeFlann(lidar_points)

        for i, tie in enumerate(tie_points.points):
            _, lidar_ind, _ = lidar_pcd_tree.search_knn_vector_3d(tie, k)
            tie_points.points[i] = lidar_points.points[lidar_ind[0]] ##TypeError: 'open3d.cpu.pybind.geometry.PointCloud' object is not subscriptable
        return tie_points

def read_lidar(path): ##Hui add
    #path1 = r'C:\Users\WANGH0M\Desktop\LiDAR\Zenmuse L1 202107 Rio Brazos.las' ##(40824215, 3)
    input_las = laspy.read(path)
    point_records = input_las.points.copy()

    # getting scaling and offset parameters
    las_scaleX = input_las.header.scale[0]
    las_offsetX = input_las.header.offset[0]
    las_scaleY = input_las.header.scale[1]
    las_offsetY = input_las.header.offset[1]
    las_scaleZ = input_las.header.scale[2]
    las_offsetZ = input_las.header.offset[2]

    # calculating coordinates
    p_X = np.array((point_records['X'] * las_scaleX) + las_offsetX)
    p_Y = np.array((point_records['Y'] * las_scaleY) + las_offsetY)
    p_Z = np.array((point_records['Z'] * las_scaleZ) + las_offsetZ)

    xyz = np.vstack((p_X, p_Y, p_Z)).transpose()
    print(xyz.shape)

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz)
    return xyz

def get_replaced_tie_points(tie_points): ##Hui add
    #path = r'C:\Users\WANGH0M\Desktop\LiDAR\Zenmuse L1 202107 Rio Brazos.las'
    path = r'C:\Users\WANGH0M\Documents\DJI\DJITerra\hui.wang.1@kaust.edu.sa\L2_Hui\lidars\terra_las\cloud_merged.las'
    lidar_points = read_lidar(path)
    tie_points = replace_tie_points_by_lidar(tie_points, lidar_points)
    return tie_points
#------------------------------------------------------------------------------------------


def get_filtered_track_structure(frame, folder, calibs):
    tie_points = frame.tie_points
    tie_points = get_replaced_tie_points(tie_points) ##Hui add

    cnt_cropped = 0

    tracks = {} # { track_id: [ point indices, good projections, bad projections ] }; projection = ( camera_key, projection_idx )
    images = {} # { camera_key: [ camera, good projections, bad projections ] }; projection = ( undistored pt in pixels, size, track_id )
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        (calib0, calib1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue

        camera_entry = [cam, [], []]

        projections = tie_points.projections[cam]
        for (i, proj) in enumerate(projections):
            track_id = proj.track_id
            if track_id not in tracks:
                tracks[track_id] = [[], [], []]

            pt = calib1.project(calib0.unproject(proj.coord))
            good = (0 <= pt.x and pt.x < calib1.width and 0 <= pt.y and pt.y < calib1.height)
            place = (1 if good else 2)

            if not good:
                cnt_cropped += 1

            pos = len(camera_entry[place])
            camera_entry[place].append((pt, proj.size, track_id))
            tracks[track_id][place].append((cam.key, pos))

        images[cam.key] = camera_entry

    for (i, pt) in enumerate(tie_points.points):
        track_id = pt.track_id
        if track_id not in tracks:
            tracks[track_id] = [[], [], []]

        tracks[track_id][0].append(i)

    print("Found", cnt_cropped, "cropped projections")
    return (tracks, images)


def save_undistorted_images(params, frame, folder, calibs):
    folder = folder + "images/"
    T = Metashape.Matrix.Diag([1, 1, 1, 1])

    cnt = 0
    for cam in frame.cameras:
        if cam.transform is None or cam.sensor is None or not cam.enabled:
            continue
        if cam.sensor.key not in calibs:
            continue
        (calib0, calib1) = get_calibs(cam, calibs)
        if calib0 is None:
            continue

        img = cam.image().warp(calib0, T, calib1, T)
        name = get_camera_name(cam)
        ext = os.path.splitext(name)[1]
        if ext.lower() in [".jpg", ".jpeg"]:
            c = Metashape.ImageCompression()
            c.jpeg_quality = params.image_quality
            img.save(folder + name, c)
        else:
            img.save(folder + name)
        cnt += 1
    print("Undistorted", cnt, "cameras")

def save_cameras(params, folder, calibs):
    use_pinhole_model = params.use_pinhole_model
    with open(folder + "sparse/0/cameras.bin", "wb") as fout:
        fout.write(u64(len(calibs)))
        for (s_key, (sensor, calib)) in calibs.items():
            fout.write(u32(s_key))
            fout.write(u32(1 if use_pinhole_model else 0))
            fout.write(u64(calib.width))
            fout.write(u64(calib.height))
            fout.write(d64(calib.f))
            if use_pinhole_model:
                fout.write(d64(calib.f))
            fout.write(d64(calib.cx + calib.width * 0.5))
            fout.write(d64(calib.cy + calib.height * 0.5))
    print("Saved", len(calibs), "calibrations")


# { camera_key: [ camera, good projections, bad projections ] }; projection = ( undistored pt in pixels, size, track_id )

def save_images(params, frame, folder, calibs, tracks, images):
    only_good = params.only_good
    T_shift = get_coord_transform(frame, params.use_localframe)

    with open(folder + "sparse/0/images.bin", "wb") as fout:
        fout.write(u64(len(images)))
        for (cam_key, [camera, good_prjs, bad_prjs]) in images.items():
            transform = T_shift * camera.transform
            R = transform.rotation().inv()
            T = -1 * (R * transform.translation())
            Q = matrix_to_quat(R)
            fout.write(u32(cam_key))
            fout.write(d64(Q.w))
            fout.write(d64(Q.x))
            fout.write(d64(Q.y))
            fout.write(d64(Q.z))
            fout.write(d64(T.x))
            fout.write(d64(T.y))
            fout.write(d64(T.z))
            fout.write(u32(camera.sensor.key))
            fout.write(bstr(get_camera_name(camera)))

            prjs = (good_prjs if only_good else good_prjs + bad_prjs)
            fout.write(u64(len(prjs)))
            for (pt, size, track_id) in prjs:
                track_id = (track_id if len(tracks[track_id][0]) != 1 else -1)
                fout.write(d64(pt.x))
                fout.write(d64(pt.y))
                fout.write(u64(track_id))
    print("Saved", len(images), "cameras")


# { track_id: [ point indices, good projections, bad projections ] }; projection = ( camera_key, projection_idx )

def save_points(params, frame, folder, calibs, tracks, images):
    only_good = params.only_good
    T = get_coord_transform(frame, params.use_localframe)
    num_pts = len(list(filter(lambda x: len(x[0]) == 1, tracks.values())))

    tie_points = frame.tie_points ##Hui add
    tie_points = get_replaced_tie_points(tie_points) ##Hui add

    with open(folder + "sparse/0/points3D.bin", "wb") as fout:
        fout.write(u64(num_pts))
        for (track_id, [points, good_prjs, bad_prjs]) in tracks.items():
            if (len(points) != 1):
                continue
            point = tie_points.points[points[0]]
            pt = T * point.coord
            track = tie_points.tracks[track_id]
            fout.write(u64(track_id))
            fout.write(d64(pt.x))
            fout.write(d64(pt.y))
            fout.write(d64(pt.z))
            fout.write(u8(track.color[0]))
            fout.write(u8(track.color[1]))
            fout.write(u8(track.color[2]))
            fout.write(d64(0))

            num = (len(good_prjs) if only_good else len(good_prjs) + len(bad_prjs))
            fout.write(u64(num))
            for (camera_key, proj_idx) in good_prjs:
                fout.write(u32(camera_key))
                fout.write(u32(proj_idx))

            if not only_good:
                for (camera_key, proj_idx) in good_prjs:
                    fout.write(u32(camera_key))
                    fout.write(u32(proj_idx + len(images[camera_key][1])))
    print("Saved", num_pts, "points from", len(tracks), "tracks")


class ExportSceneParams():
    def __init__(self):
        # default values for parameters
        self.all_chunks = False
        self.all_frames = False

        self.zero_cxy = True
        self.use_localframe = True
        self.image_quality = 90
        self.confirm_deletion = True
        self.use_pinhole_model = True
        self.only_good = True

    def log(self):
        print("All chunks:", self.all_chunks)
        print("All frames:", self.all_frames)
        print("Zero cx and cy:", self.zero_cxy)
        print("Use local coordinate frame:", self.use_localframe)
        print("Image quality:", self.image_quality)
        print("Confirm deletion:", self.confirm_deletion)
        print("Using pinhole model instead of simple_pinhole:", self.use_pinhole_model)
        print("Using only uncropped projections:", self.only_good)


def export_for_gaussian_splatting(params = ExportSceneParams(), progress = QtWidgets.QProgressBar()):
    log_result = lambda x: print("", x, "-----------------------------------", sep="\n")
    progress.setMinimum(0)
    progress.setMaximum(1000)
    set_progress = lambda x: progress.setValue(int(x * 1000))
    params.log()

    folder = Metashape.app.getExistingDirectory("Output folder")
    if len(folder) == 0:
        log_result("No chosen folder")
        return
    folder = folder + "/"
    print(folder)

    chunk_dirs = get_chunk_dirs(folder, params)
    if len(chunk_dirs) == 0:
        log_result("Aborted")
        return

    chunk_num = len(chunk_dirs)
    for chunk_id, (chunk_key, chunk_folder) in enumerate(chunk_dirs.items()):
        chunk = [ck for ck in Metashape.app.document.chunks if ck.key == chunk_key]
        if (len(chunk) != 1):
            print("Chunk not found, key =", chunk_key)
            continue
        chunk = chunk[0]

        frame_num = len(chunk.frames) if params.all_frames else 1
        prog_step = 1 / chunk_num
        set_progress(prog_step * chunk_id)
        set_progress_frame = lambda n: set_progress(prog_step * (chunk_id + n / frame_num))
        frame_cnt = 0

        for frame_id, frame in enumerate(chunk.frames):
            if not frame.tie_points:
                continue
            if not params.all_frames and not (frame == chunk.frame):
                continue
            set_progress_frame(frame_cnt)
            frame_cnt += 1

            folder = chunk_folder + ("" if frame_num == 1 else "frame_" + str(frame_id).zfill(6) + "/")
            print("\n" + folder)

            if not build_dir_structure(folder, params.confirm_deletion):
                log_result("Aborted")
                return

            calibs = compute_undistorted_calibs(frame, params.zero_cxy)
            (tracks, images) = get_filtered_track_structure(frame, folder, calibs)

            save_undistorted_images(params, frame, folder, calibs)
            save_cameras(params, folder, calibs)
            save_images(params, frame, folder, calibs, tracks, images)
            save_points(params, frame, folder, calibs, tracks, images)

    set_progress(1)
    log_result("Done")

class ExportSceneGUI(QtWidgets.QDialog):

    def run_export(self):
        for button in self.buttons:
            button.setEnabled(False)

        params = ExportSceneParams()
        params.all_chunks = self.radioBtn_allC.isChecked()
        params.all_frames = self.radioBtn_allF.isChecked()
        params.zero_cxy = self.zcxyBox.isChecked()
        params.use_localframe = self.locFrameBox.isChecked()
        params.image_quality = self.imgQualSpBox.value()
        try:
            export_for_gaussian_splatting(params, self.pBar)
        finally:
            self.done(0)

    def __init__(self, parent):
        QtWidgets.QDialog.__init__(self, parent)
        self.setWindowTitle("Export scene in Colmap format:")

        defaults = ExportSceneParams()

        self.btnQuit = QtWidgets.QPushButton("Quit")
        self.btnQuit.setFixedSize(100,25)

        self.btnP1 = QtWidgets.QPushButton("Export")
        self.btnP1.setFixedSize(100,25)

        self.pBar = QtWidgets.QProgressBar()
        self.pBar.setTextVisible(False)
        self.pBar.setFixedSize(100, 25)

        self.chnkTxt = QtWidgets.QLabel()
        self.chnkTxt.setText("Chunks:")
        self.chnkTxt.setFixedSize(100, 25)

        self.frmsTxt = QtWidgets.QLabel()
        self.frmsTxt.setText("Frames:")
        self.frmsTxt.setFixedSize(100, 25)

        self.chunk_group = QtWidgets.QButtonGroup()
        self.radioBtn_allC = QtWidgets.QRadioButton("all chunks")
        self.radioBtn_selC = QtWidgets.QRadioButton("selected")
        self.chunk_group.addButton(self.radioBtn_selC)
        self.chunk_group.addButton(self.radioBtn_allC)
        self.radioBtn_allC.setChecked(defaults.all_chunks)
        self.radioBtn_selC.setChecked(not defaults.all_chunks)

        self.frames_group = QtWidgets.QButtonGroup()
        self.radioBtn_allF = QtWidgets.QRadioButton("all frames")
        self.radioBtn_selF = QtWidgets.QRadioButton("active")
        self.frames_group.addButton(self.radioBtn_selF)
        self.frames_group.addButton(self.radioBtn_allF)
        self.radioBtn_allF.setChecked(defaults.all_frames)
        self.radioBtn_selF.setChecked(not defaults.all_frames)

        self.zcxyTxt = QtWidgets.QLabel()
        self.zcxyTxt.setText("Enforce zero cx, cy")
        self.zcxyTxt.setFixedSize(100, 25)

        self.zcxyBox = QtWidgets.QCheckBox()
        self.zcxyBox.setChecked(defaults.zero_cxy)

        self.locFrameTxt = QtWidgets.QLabel()
        self.locFrameTxt.setText("Use localframe")
        self.locFrameTxt.setFixedSize(100, 25)

        self.locFrameBox = QtWidgets.QCheckBox()
        self.locFrameBox.setChecked(defaults.use_localframe)

        self.imgQualTxt = QtWidgets.QLabel()
        self.imgQualTxt.setText("Image quality")
        self.imgQualTxt.setFixedSize(100, 25)

        self.imgQualSpBox = QtWidgets.QSpinBox()
        self.imgQualSpBox.setMinimum(0)
        self.imgQualSpBox.setMaximum(100)
        self.imgQualSpBox.setValue(defaults.image_quality)


        zcxyToolTip = "Output camera calibrations will have zero cx and cy\nMay result in information loss during export (large cropping)\nShould be checked until Gaussian Splatting software considers this parameters"
        self.zcxyTxt.setToolTip(zcxyToolTip)
        self.zcxyBox.setToolTip(zcxyToolTip)

        locFrameToolTip = "Shifts coordinates origin to the center of the bounding box\nUses localframe rotation at this point\nThis is useful to fix large coordinates"
        self.locFrameTxt.setToolTip(locFrameToolTip)
        self.locFrameBox.setToolTip(locFrameToolTip)

        imgQualToolTip = "Quality of the output undistorted images (jpeg only)\nMin = 0, Max = 100"
        self.imgQualTxt.setToolTip(imgQualToolTip)
        self.imgQualSpBox.setToolTip(imgQualToolTip)


        layout = QtWidgets.QGridLayout()
        layout.setSpacing(9)
        layout.addWidget(self.chnkTxt, 1, 0)
        layout.addWidget(self.radioBtn_allC, 1, 1)
        layout.addWidget(self.radioBtn_selC, 1, 2)
        layout.addWidget(self.frmsTxt, 2, 0)
        layout.addWidget(self.radioBtn_allF, 2, 1)
        layout.addWidget(self.radioBtn_selF, 2, 2)
        layout.addWidget(self.zcxyTxt, 3, 0)
        layout.addWidget(self.zcxyBox, 3, 1)
        layout.addWidget(self.locFrameTxt, 4, 0)
        layout.addWidget(self.locFrameBox, 4, 1)
        layout.addWidget(self.imgQualTxt, 5, 0)
        layout.addWidget(self.imgQualSpBox, 5, 1, 1, 2)
        layout.addWidget(self.pBar, 6, 0)
        layout.addWidget(self.btnP1, 6, 1)
        layout.addWidget(self.btnQuit, 6, 2)
        self.setLayout(layout)

        self.buttons = [self.btnP1, self.btnQuit, self.radioBtn_allC, self.radioBtn_selC, self.radioBtn_allF, self.radioBtn_selF, self.zcxyBox, self.locFrameBox, self.imgQualSpBox]

        proc = lambda : self.run_export()

        QtCore.QObject.connect(self.btnP1, QtCore.SIGNAL("clicked()"), proc)
        QtCore.QObject.connect(self.btnQuit, QtCore.SIGNAL("clicked()"), self, QtCore.SLOT("reject()"))

        self.exec()


def export_for_gaussian_splatting_gui():
    global app
    app = QtWidgets.QApplication.instance()
    parent = app.activeWindow()
    dlg = ExportSceneGUI(parent)

label = "Scripts/Export Colmap project (for Gaussian Splatting)"
Metashape.app.addMenuItem(label, export_for_gaussian_splatting_gui)
print("To execute this script press {}".format(label))

# If you want to run this script automatically - use this instead:
#export_for_gaussian_splatting()
