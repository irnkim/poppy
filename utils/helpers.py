import torch, os, warnings, numpy as np, torch.nn.functional as F
from PIL import Image
import csv
import imageio

warnings.filterwarnings("ignore")

eps = torch.finfo(torch.float32).eps


def polarization_physics_loss_Lopt(s_obs, s_hat):
    L_s0 = F.l1_loss(s_hat[0], s_obs[0])
    L_s1 = F.l1_loss(s_hat[1], s_obs[1])
    L_s2 = F.l1_loss(s_hat[2], s_obs[2])
    return L_s0 + L_s1 + L_s2


def calculate_observed_polarization(I_0, I_pi_4, I_pi_2, I_3pi_4):
    I_obs = I_0 + I_pi_2
    Q_obs = I_0 - I_pi_2
    U_obs = I_pi_4 - I_3pi_4
    phi_obs = 0.5 * np.arctan2(U_obs, Q_obs)
    phi_obs = np.remainder(phi_obs + np.pi / 2, np.pi) - np.pi / 2
    phi_obs = np.clip(phi_obs, -np.pi / 2, np.pi / 2)
    rho_obs = np.sqrt(Q_obs**2 + U_obs**2) / (I_obs + 1e-8)
    rho_obs = np.nan_to_num(rho_obs, nan=0.0)
    return I_obs, Q_obs, U_obs, phi_obs.mean(axis=2), rho_obs.mean(axis=2)


def read_img(path):
    img = imageio.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    depth = np.iinfo(img.dtype).max
    img_norm = img.astype(np.float32) / depth
    return img_norm


def read_mask(path):
    if os.path.isfile(path):
        ext = path.split(".")[-1]
        if ext in ["png", "jpg", "tiff"]:
            mask = read_img(path)
        elif path.endswith(".npy"):
            mask = np.load(path)
        else:
            raise ValueError(f"Invalid file extension")
        mask = (mask > 0.5).astype(bool)
    else:
        print(f'Mask file "{path}" doesnot exsit')
        mask = None
    return mask


def read_normal(path):
    ext = path.split(".")[-1].lower()
    if ext in ["png", "jpg", "jpeg", "tiff"]:
        n_img = read_img(path)
        normal = n_img * 2.0 - 1.0
    elif ext == "npy":
        normal = np.load(path)
    else:
        raise ValueError(f"Unsupported normal file type: {path}")
    if normal.shape[-1] == 4:
        normal = normal[..., :3]
    mag = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-8
    return normal / mag


def compute_metrics(normal_pred, normal_gt, mask, eps=1e-12):
    normal_pred = normal_pred / (np.linalg.norm(normal_pred, axis=-1, keepdims=True) + eps)
    normal_gt = normal_gt / (np.linalg.norm(normal_gt, axis=-1, keepdims=True) + eps)
    cos = (normal_pred * normal_gt).sum(axis=-1).clip(-1, 1)
    errmap = np.degrees(np.arccos(cos))
    if mask is not None:
        mask = mask.squeeze()[0].cpu().numpy()
        mask = (mask > 0.5).astype(bool)
        angle_err = errmap[mask].flatten()
    else:
        angle_err = errmap.flatten()
    return dict(
        mean=angle_err.mean(),
        median=np.median(angle_err),
        rmse=np.sqrt((angle_err**2).mean()),
        acc_11=(angle_err < 11.25).mean(),
        acc_22=(angle_err < 22.5).mean(),
        acc_30=(angle_err < 30.0).mean(),
    )


def normal_vis(pipe, pred, mask, convert=True):
    if pred is None:
        return None
    if pipe is None or not hasattr(pipe, "image_processor"):
        if torch.is_tensor(pred):
            pred = pred.squeeze().cpu().numpy().transpose(1, 2, 0)
        vis = (pred + 1.0) * 0.5 * 255.0
    else:
        vis = pipe.image_processor.visualize_normals(pred)[0]
    mask = mask.squeeze().cpu().numpy().transpose(1, 2, 0)
    vis_np = np.array(vis) * mask
    return Image.fromarray(vis_np.astype(np.uint8)), vis_np.astype(np.float32) / 255.0


def save_Ls_Ld_vis(vis_dir, image_name, Ls, Ld):
    for name, val in (("Ls", Ls), ("Ld", Ld)):
        val_img = Image.fromarray((np.clip(val, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))
        val_img.save(os.path.join(vis_dir, f"{image_name}_{name}.png"))


def open_log_file(out_dir):
    loss_log_path = os.path.join(out_dir, "log.csv")
    f = open(loss_log_path, "w", newline="")
    writer_csv = csv.writer(f)
    writer_csv.writerow(["object", "time_sec", "mean_loss", "median", "rmse", "acc_11", "acc_22", "acc_30"])
    print("Logging to:", loss_log_path, "\n")
    return f, writer_csv


def fresnel_R_perp_par(theta, eta_i=1.0, eta_t=1.5, eps=1e-6):
    sin_i = torch.sin(theta)
    sin_t = (eta_i / eta_t) * sin_i
    cos_i = torch.cos(theta)
    cos_t = torch.sqrt(1.0 - sin_t**2 + eps)
    r_perp = (eta_i * cos_i - eta_t * cos_t) / (eta_i * cos_i + eta_t * cos_t + eps)
    r_par = (eta_t * cos_i - eta_i * cos_t) / (eta_t * cos_i + eta_i * cos_t + eps)
    return r_perp**2, r_par**2


def dolp_diffuse(theta, eta=1.5, eps=1e-6):
    s2 = torch.sin(theta) ** 2
    c = torch.cos(theta)
    term = torch.sqrt(eta**2 - s2 + eps)
    num = (eta - 1.0 / eta) ** 2 * s2
    den = 2 + 2 * eta**2 - (eta + 1.0 / eta) ** 2 * s2 + 4 * c * term
    return num / (den + eps)


def get_stokes_Lopt(Ls, Ld, alpha, theta, eta=1.5, eps=1e-6):
    phi_s = alpha + torch.pi / 2
    phi_s = torch.remainder(phi_s + torch.pi / 2, torch.pi) - torch.pi / 2
    phi_d = alpha

    R_perp, R_par = fresnel_R_perp_par(theta, 1.0, eta, eps)
    rho_s = (R_perp - R_par) / (R_perp + R_par + eps)
    rho_d = dolp_diffuse(theta, eta, eps)

    Is, Id = Ls, Ld
    Q = Id * rho_d * torch.cos(2 * phi_d) + Is * rho_s * torch.cos(2 * phi_s)
    U = Id * rho_d * torch.sin(2 * phi_d) + Is * rho_s * torch.sin(2 * phi_s)
    I = Id + Is

    stokes = torch.stack([I, Q, U], dim=0)
    dolp = torch.sqrt(Q**2 + U**2 + eps) / (I + eps)
    aolp = 0.5 * torch.atan2(U, Q + eps)
    return stokes, aolp, dolp


def normal2polar_clip(image, normal, fov, Ls, Ld, eta=1.5, eps=eps, tau=1e-2):
    n = normal / (torch.linalg.norm(normal, dim=1, keepdim=True) + eps)
    nx = n[:, 0, :, :]
    ny = n[:, 1, :, :]
    nz = n[:, 2, :, :]
    theta = torch.acos(nz.float().clamp(-1 + 1e-6, 1 - 1e-6)).to(nz.dtype)
    psi = torch.atan2(ny, nx + eps)
    Ls, Ld = F.relu(Ls), F.relu(Ld)
    stokes, aolp, dolp = get_stokes_Lopt(Ls, Ld, psi, theta, eta, eps)
    return stokes, aolp, dolp


def valid_mask_from_s0_both(
    img: torch.Tensor,
    mask,
    rho_obs,
    dark_thr: float = 0.01,
    sat_thr: float = 0.99,
):
    I = img.mean(dim=1, keepdim=True)
    rho_mask = (rho_obs <= 1.0) * mask
    sat_mask = (img.mean(dim=1, keepdim=True) <= 1.0) * mask
    dark_mask = I > dark_thr
    return rho_mask & sat_mask & dark_mask
