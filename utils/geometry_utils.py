import mcubes
import numpy as np
import torch


def find_surface_points(sdf, d_all, device="cuda"):
    # interpolate SDF zero-crossing points
    # shape of sdf and d_all: only inside  [B, N_rays, N_samples+N_importance]
    sdf_bool_1 = (
        sdf[..., 1:] * sdf[..., :-1] < 0
    )  # [B, N_rays, N_samples+N_importance-1]
    # only find backward facing surface points, not forward facing
    sdf_bool_2 = sdf[..., 1:] < sdf[..., :-1]
    sdf_bool = torch.logical_and(
        sdf_bool_1, sdf_bool_2
    )  # [B, N_rays, N_samples+N_importance-1]
    # [B, N_rays]
    max, max_indices = torch.max(sdf_bool, dim=2)
    network_mask = max > 0
    d_surface = torch.zeros_like(network_mask, device=device).float()  # [B, N_rays]
    sdf_0 = torch.gather(
        sdf[network_mask], 1, max_indices[network_mask][..., None]
    ).squeeze()  # [N_masked_rays]
    sdf_1 = torch.gather(
        sdf[network_mask], 1, max_indices[network_mask][..., None] + 1
    ).squeeze()  # [N_masked_rays]
    d_0 = torch.gather(
        d_all[network_mask], 1, max_indices[network_mask][..., None]
    ).squeeze()  # [N_masked_rays]
    d_1 = torch.gather(
        d_all[network_mask], 1, max_indices[network_mask][..., None] + 1
    ).squeeze()  # [N_masked_rays]
    d_surface[network_mask] = (sdf_0 * d_1 - sdf_1 * d_0) / (
        sdf_0 - sdf_1
    )  # [N_masked_rays]
    return d_surface, network_mask  # [B, N_rays]


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat(
                        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)],
                        dim=-1,
                    )
                    val = (
                        query_func(pts)
                        .reshape(len(xs), len(ys), len(zs))
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    u[
                        xi * N : xi * N + len(xs),
                        yi * N : yi * N + len(ys),
                        zi * N : zi * N + len(zs),
                    ] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print("threshold: {}".format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = (
        vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :]
        + b_min_np[None, :]
    )
    return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples
        )
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples
