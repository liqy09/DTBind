import os
import pickle
import random
import numpy as np
from sklearn.neighbors import KDTree
from numpy.linalg import norm
import pymeshlab
from subprocess import Popen, PIPE

# 手动设置 bin_path
bin_path = {
    'MSMS': './dMaSIF-master/MSMS/msms.x86_64Linux2.2.6.1'
}

def read_msms(file_root):
    vertfile = open(file_root + ".vert")
    meshdata = (vertfile.read().rstrip()).split("\n")
    vertfile.close()

    count = {}
    header = meshdata[2].split()
    count["vertices"] = int(header[0])
    vertices = np.zeros((count["vertices"], 3))
    normalv = np.zeros((count["vertices"], 3))
    atom_id = [""] * count["vertices"]
    res_id = [""] * count["vertices"]
    for i in range(3, len(meshdata)):
        fields = meshdata[i].split()
        vi = i - 3
        vertices[vi][0] = float(fields[0])
        vertices[vi][1] = float(fields[1])
        vertices[vi][2] = float(fields[2])
        normalv[vi][0] = float(fields[3])
        normalv[vi][1] = float(fields[4])
        normalv[vi][2] = float(fields[5])
        atom_id[vi] = fields[7]
        res_id[vi] = fields[9]
        count["vertices"] -= 1

    facefile = open(file_root + ".face")
    meshdata = (facefile.read().rstrip()).split("\n")
    facefile.close()

    header = meshdata[2].split()
    count["faces"] = int(header[0])
    faces = np.zeros((count["faces"], 3), dtype=int)

    for i in range(3, len(meshdata)):
        fi = i - 3
        fields = meshdata[i].split()
        faces[fi][0] = int(fields[0]) - 1
        faces[fi][1] = int(fields[1]) - 1
        faces[fi][2] = int(fields[2]) - 1
        count["faces"] -= 1

    assert count["vertices"] == 0
    assert count["faces"] == 0

    return vertices, faces, normalv, res_id


def computeMSMS(pair, dir_opts, probe_radius):
    xyzrn = os.path.join(dir_opts['xyzrn_dir'], pair + '.xyzrn')
    msms_file_base = os.path.join(dir_opts['msms_dir'], pair + str(random.randint(1, 10000000)))
    if not os.path.exists(dir_opts['msms_dir']):
        os.makedirs(dir_opts['msms_dir'])

    msms_bin = bin_path['MSMS']
    FNULL = open(os.devnull, 'w')
    args = [msms_bin, "-density", "3", "-hdensity", "3", "-probe",
            str(probe_radius), "-if", xyzrn, "-of", msms_file_base, "-af", msms_file_base]

    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    if p2.returncode != 0:
        print(f"MSMS failed for {pair} with error: {stderr.decode('utf-8')}")
        return None, None, None, None

    vertices, faces, normalv, res_id = read_msms(msms_file_base)
    return vertices, faces, normalv, res_id


def fix_mesh(mesh, vertice_info, resolution, detail="normal"):
    bbox_min = mesh.bounding_box().min()
    bbox_max = mesh.bounding_box().max()
    diag_len = norm(bbox_max - bbox_min)
    if detail == "normal":
        target_len = diag_len * 5e-3
    elif detail == "high":
        target_len = diag_len * 2.5e-3
    elif detail == "low":
        target_len = diag_len * 1e-2

    target_len = resolution

    ms = pymeshlab.MeshSet()
    ms.add_mesh(mesh)
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(ms.current_mesh().vertex_number() * target_len))
    ms.meshing_remove_connected_component_by_diameter(mincomponentdiag=pymeshlab.AbsoluteValue(target_len))
    ms.meshing_merge_close_vertices(threshold=pymeshlab.AbsoluteValue(target_len))

    new_mesh = ms.current_mesh()

    new_vertice_info = []
    kdtree = KDTree(mesh.vertex_matrix())
    for vertex in new_mesh.vertex_matrix():
        dis, pos = kdtree.query(vertex[None, :])
        new_vertice_info.append(vertice_info[pos[0, 0]])

    return new_mesh.vertex_matrix(), new_mesh.face_matrix(), new_vertice_info


def compute_normal(vertices, faces):
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertices, faces)
    ms.add_mesh(mesh)
    ms.apply_filter('compute_normal_for_point_clouds')
    normals = ms.current_mesh().vertex_normal_matrix()
    return normals


def save_ply(
        filename,
        vertices,
        faces=[],
        normals=None,
        u=None,
        v=None,
        charges=None,
        vertex_cb=None,
        hbond=None,
        hphob=None,
        iface_residue=None,
        iface_atom=None,
        iface_vertex=None,
        normalize_charges=False,
        features=None,
        label=None,
        curvature=None,
        patch=None
):
    ms = pymeshlab.MeshSet()
    mesh = pymeshlab.Mesh(vertices, faces)
    ms.add_mesh(mesh)
    if normals is not None:
        ms.current_mesh().vertex_normal_matrix()[:] = normals
    if u is not None:
        ms.set_vertex_attribute("vertex_ux", u[:, 0])
        ms.set_vertex_attribute("vertex_uy", u[:, 1])
        ms.set_vertex_attribute("vertex_uz", u[:, 2])
    if v is not None:
        ms.set_vertex_attribute("vertex_vx", v[:, 0])
        ms.set_vertex_attribute("vertex_vy", v[:, 1])
        ms.set_vertex_attribute("vertex_vz", v[:, 2])
    if charges is not None:
        if normalize_charges:
            charges = charges / 10
        ms.set_vertex_attribute("charge", charges)
    if hbond is not None:
        ms.set_vertex_attribute("hbond", hbond)
    if vertex_cb is not None:
        ms.set_vertex_attribute("vertex_cb", vertex_cb)
    if hphob is not None:
        ms.set_vertex_attribute("vertex_hphob", hphob)
    if iface_residue is not None:
        ms.set_vertex_attribute("vertex_iface_residue", iface_residue)
    if iface_atom is not None:
        ms.set_vertex_attribute("vertex_iface_atom", iface_atom)
    if iface_vertex is not None:
        ms.set_vertex_attribute("vertex_iface_vertex", iface_vertex)
    if features is not None:
        ms.set_vertex_attribute("features", features)
    if label is not None:
        ms.set_vertex_attribute("label", label)
    if curvature is not None:
        ms.set_vertex_attribute('curvature', curvature)
    if patch is not None:
        ms.set_vertex_attribute("patch", patch)
    ms.save_current_mesh(filename, binary=False)


def process_pdb_files(input_dir, output_dir):
    """
    处理提取的特征文件并保存为 PLY 文件。

    Args:
        input_dir (str): 特征文件所在目录。
        output_dir (str): 输出文件所在目录。
    """
    dir_opts = {
        'xyzrn_dir': os.path.join(input_dir, 'xyzrn'),
        'msms_dir': os.path.join(output_dir, 'msms')
    }

    with open(os.path.join(input_dir, 'xyzrn_data.pkl'), 'rb') as f:
        xyzrn_data = pickle.load(f)

    for pair, xyzrn_lines in xyzrn_data.items():
        try:
            # 计算 MSMS 表面
            vertices, faces, normalv, res_id = computeMSMS(pair, dir_opts, probe_radius=1.5)

            if vertices is None or faces is None:
                print(f"Skipping {pair} due to MSMS failure.")
                continue

            # 创建和修复网格
            mesh = pymeshlab.Mesh(vertices, faces)
            vertices, faces, new_vertice_info = fix_mesh(mesh, res_id, resolution=1.2)

            # 计算法向量
            normals = compute_normal(vertices, faces)

            # 保存 PLY 文件
            ply_file = os.path.join(output_dir, pair + '.ply')
            save_ply(ply_file, vertices, faces, normals=normals)

        except Exception as e:
            print(f"Error processing {pair}: {e}")
            continue


# 示例使用
input_dir = './data_process/mesh/biosnap_pro'
output_dir = './data_process/mesh/biosnap_pro/ply'
process_pdb_files(input_dir, output_dir)

