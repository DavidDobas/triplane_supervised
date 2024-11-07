import pandas as pd
import json
import os
import zipfile
import numpy as np
import cv2

path = '/home/anya/Programs/eg3d/eg3d/datasets/'
zip_name = 'chest_rot.zip'
fname = os.path.join(path, zip_name)

def excel_from_json(fname: str, path: str):
    with zipfile.ZipFile(fname, 'r') as zip_file:
        with zip_file.open('dataset.json', 'r') as file:
            data = json.load(file)
            # 'data' is a list of dictionaries, where each dictionary contains an image ID and its matrix
            # Example of data structure: {'labels': [[...], ...]}
            with pd.ExcelWriter(os.path.join(path, 'shapenet.xlsx'), engine='xlsxwriter') as writer:
                rows = []
                for item in data['labels']:
                    row = [item[0]] + item[1]
                    rows.append(row)

                column_names = ['image ID'] + [f'val {i}' for i in range(1, max(len(row) for row in rows))]
                df = pd.DataFrame(rows, columns=column_names)
                df.to_excel(writer)

# convert json to json, but separately for each image for extrinsic visualisation
def json_separate(fname: str, path: str, zip_name: str):
    matrix = {}
    new_path = os.path.join(path, f'{zip_name[:-4]}/trajectory')
    os.makedirs(new_path, exist_ok=True)
    with zipfile.ZipFile(fname, 'r') as zip_file:
        with zip_file.open('dataset.json', 'r') as file:
            data = json.load(file)
            for item in data['labels']:
                image_id = item[0]
                matrix['extrinsic matrix'] = item[1][:16]
                name = f'{image_id[9:17]}.json'
                with open(os.path.join(new_path, name), 'w') as file:
                    json.dump(matrix, file, indent=2)

def generate_extrinsic_matrix(rotation_deg, elevation_deg):
    # Convert degrees to radians
    rotation_rad = np.deg2rad(rotation_deg)
    elevation_rad = np.deg2rad(elevation_deg)

    # # Rotation matrix around the Y-axis
    # R_y = np.array([[np.cos(rotation_rad), 0, np.sin(rotation_rad)],
    #                 [0, 1, 0],
    #                 [-np.sin(rotation_rad), 0, np.cos(rotation_rad)]])
    
    # Rotation matrix around the X-axis for elevation
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(elevation_rad), -np.sin(elevation_rad)],
                    [0, np.sin(elevation_rad), np.cos(elevation_rad)]])
    
    # Rotation matrix around the Z-axis
    R_z = np.array([[np.cos(rotation_rad), -np.sin(rotation_rad), 0],
                    [np.sin(rotation_rad), np.cos(rotation_rad), 0],
                    [0, 0, 1]])
    
    # Combined rotation matrix
    # R = R_x.dot(R_y) # Assuming the rotation order is X then Y
    R = np.dot(R_z, R_x) # Assuming the rotation order is Z then X

    # Define the translation vector (example: placing the camera at the origin)
    t = np.array([[0], [0], [0]]) # Adjust based on the specific camera position
    
    # Combine R and t to form the extrinsic matrix
    E = np.hstack((R, t))
    E = np.vstack((E, [0, 0, 0, 1])) # Add the last row [0, 0, 0, 1] to make it 4x4
    
    return E

def generate_rotation(path: str, zip_name: str):
    elevation_deg = 90
    i = 0

    for rotation_deg in range(0, 360, 8):
        extrinsic_matrix = generate_extrinsic_matrix(rotation_deg, elevation_deg)
        matrix = {}
        new_path = os.path.join(path, f'{zip_name[:-4]}/trajectory')
        os.makedirs(new_path, exist_ok=True)
        matrix['extrinsic matrix'] = extrinsic_matrix.flatten().tolist() 
        name = f'{i:08d}.json'
        with open(os.path.join(new_path, name), 'w') as file:
            json.dump(matrix, file, indent=2)
        i += 1

# From MedNeRF
def get_render_poses(radius, angle_range=(0, 360), theta=20, N=72, swap_angles=False):
    poses = []
    theta = max(0.1, theta)
    for angle in np.linspace(angle_range[0],angle_range[1],N+1)[:-1]:
        angle = max(0.1, angle)
        if swap_angles:
            loc = polar_to_cartesian(radius, theta, angle, deg=True)
        else:
            loc = polar_to_cartesian(radius, angle, theta, deg=True)
        R = look_at(loc)[0]
        RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
        RT = np.concatenate([RT, np.array([[0, 0, 0, 1]])], axis=0)
        poses.append(RT)
    return np.stack(poses)

def polar_to_cartesian(r, theta, phi, deg=True):
    if deg:
        phi = phi * np.pi / 180
        theta = theta * np.pi / 180
    cx = np.sin(phi) * np.cos(theta)
    cy = np.sin(phi) * np.sin(theta)
    cz = np.cos(phi)
    return r * np.stack([cx, cy, cz])

def look_at(eye, at=np.array([0, 0, 0]), up=np.array([0, 0, 1]), eps=1e-5):
    at = at.astype(float).reshape(1, 3)
    up = up.astype(float).reshape(1, 3)

    eye = eye.reshape(-1, 3)
    up = up.repeat(eye.shape[0] // up.shape[0], axis=0)
    eps = np.array([eps]).reshape(1, 1).repeat(up.shape[0], axis=0)

    z_axis = eye - at
    z_axis /= np.max(np.stack([np.linalg.norm(z_axis, axis=1, keepdims=True), eps]))

    x_axis = np.cross(up, z_axis)
    x_axis /= np.max(np.stack([np.linalg.norm(x_axis, axis=1, keepdims=True), eps]))

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.max(np.stack([np.linalg.norm(y_axis, axis=1, keepdims=True), eps]))

    r_mat = np.concatenate((x_axis.reshape(-1, 3, 1), y_axis.reshape(-1, 3, 1), z_axis.reshape(-1, 3, 1)), axis=2)

    return r_mat

def generate_rotation_m(path: str, zip_name: str):
    "Function generates extrinsic matrices for 360 degree rotation around the object with a fixed elevation angle and a given radius"
    elevation_deg = 45
    i = 0

    extrinsic_matrix = get_render_poses(radius=10.5, theta=elevation_deg)
    matrix = {}
    new_path = os.path.join(path, f'{zip_name[:-4]}/pose')
    os.makedirs(new_path, exist_ok=True)
    for i in range(len(extrinsic_matrix)):
        matrix['extrinsic matrix'] = extrinsic_matrix[i].flatten().tolist() 
        name = f'{i:06d}.json'
        with open(os.path.join(new_path, name), 'w') as file:
            json.dump(matrix, file, indent=2)



#excel_from_json(fname, path)
json_separate(fname, path, zip_name)
#generate_rotation_m(path, zip_name)