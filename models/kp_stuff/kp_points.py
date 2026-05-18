import numpy as np
import torch


def create_3D_rotations(axis, angle):
    """
    Create rotation matrices from a list of axes and angles. Code from wikipedia on quaternions
    :param axis: float32[N, 3]
    :param angle: float32[N,]
    :return: float32[N, 3, 3]
    """

    t1 = np.cos(angle)
    t2 = 1 - t1
    t3 = axis[:, 0] * axis[:, 0]
    t6 = t2 * axis[:, 0]
    t7 = t6 * axis[:, 1]
    t8 = np.sin(angle)
    t9 = t8 * axis[:, 2]
    t11 = t6 * axis[:, 2]
    t12 = t8 * axis[:, 1]
    t15 = axis[:, 1] * axis[:, 1]
    t19 = t2 * axis[:, 1] * axis[:, 2]
    t20 = t8 * axis[:, 0]
    t24 = axis[:, 2] * axis[:, 2]
    R = np.stack(
        [
            t1 + t2 * t3,
            t7 - t9,
            t11 + t12,
            t7 + t9,
            t1 + t2 * t15,
            t19 - t20,
            t11 - t12,
            t19 + t20,
            t1 + t2 * t24,
        ],
        axis=1,
    )

    return np.reshape(R, (-1, 3, 3))


@torch.no_grad()
def shell_kernel_generator(radius, shell_n_pts, num_kernels=1, dimension=3):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param shell_n_pts: list of the number of points per shell
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    n_shell = len(shell_n_pts)
    assert n_shell > 1
    assert shell_n_pts[0] == 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2
    shell_l = diameter0 / (2 * n_shell - 1)
    shell_radiuses = [s * shell_l for s in range(n_shell)]
    total_points = int(np.sum(shell_n_pts))

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Add center points
    kernel_points = torch.zeros((num_kernels, 1, dimension))

    # Add random shell points
    for num_points, shell_r in zip(shell_n_pts[1:], shell_radiuses[1:]):

        if dimension == 2:
            theta = torch.rand(num_kernels, num_points) * 2 * np.pi
            u = torch.stack([torch.cos(theta), torch.sin(theta)], dim=2)

        elif dimension == 3:
            theta = torch.rand(num_kernels, num_points) * 2 * np.pi
            phi = (torch.rand(num_kernels, num_points) - 0.5) * np.pi
            u = torch.stack(
                [
                    torch.cos(theta) * torch.cos(phi),
                    torch.sin(theta) * torch.cos(phi),
                    torch.sin(phi),
                ],
                axis=2,
            )

        else:
            raise ValueError("Unsupported dimension for shelled kernel generation")
        kernel_points = torch.cat((kernel_points, u * shell_r), dim=1)

    kernel_points = kernel_points.to(device)

    #####################
    # Kernel optimization
    #####################

    n_iter = 10000
    saved_gradient_norms = []
    old_gradient_norms = kernel_points.new_zeros((num_kernels, total_points))
    for iter in range(n_iter):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = kernel_points.unsqueeze(2)
        B = kernel_points.unsqueeze(1)
        interd2 = torch.sum(torch.pow(A - B, 2), dim=-1)
        inter_grads = (A - B) / (torch.pow(interd2.unsqueeze(-1), 3 / 2) + 1e-6)
        gradients = torch.sum(inter_grads, dim=1)

        # Reduce gradients to tangential components (nk, K, 3)
        normals = kernel_points / (
            torch.linalg.norm(kernel_points, dim=-1, keepdims=True) + 1e-6
        )
        gradients -= torch.sum(gradients * normals, dim=-1, keepdims=True) * normals

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = torch.sqrt(torch.sum(torch.pow(gradients, 2), dim=-1))
        saved_gradient_norms.append(torch.max(gradients_norms, dim=1)[0])

        # Stop if all moving points are gradients fixed (low gradients diff)

        if (
            torch.max(
                torch.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])
            ).item()
            < thresh
        ):
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = torch.clamp(moving_factor * gradients_norms, max=clip)

        # Move points
        kernel_points -= (
            moving_dists.unsqueeze(-1)
            * gradients
            / (gradients_norms.unsqueeze(-1) + 1e-6)
        )

        # Readjust radiuses to remain on the shell
        i0 = 0
        for n_p, shell_r in zip(shell_n_pts, shell_radiuses):
            kernel_points[:, i0 : i0 + n_p] *= shell_r / (
                torch.linalg.norm(
                    kernel_points[:, i0 : i0 + n_p], dim=-1, keepdims=True
                )
                + 1e-6
            )
            i0 += n_p

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale kernels with real radius
    kernel_points = (kernel_points * radius).cpu().numpy()
    saved_gradient_norms = torch.stack(saved_gradient_norms).cpu().numpy()

    return kernel_points, saved_gradient_norms


def load_kernels(radius, shell_sizes, dimension, fixed):
    """
    shell_sizes is kpoints!
    """

    # Create kernels
    kernel_points, grad_norms = shell_kernel_generator(
        1.0,
        shell_sizes,
        num_kernels=100,
        dimension=dimension,
    )

    # Find best candidate
    best_k = np.argmin(grad_norms[-1, :])

    # Save points
    kernel_points = kernel_points[best_k, :, :]

    # Random roations for the kernel
    # N.B. 4D random rotations not supported yet
    R = np.eye(dimension)
    theta = np.random.rand() * 2 * np.pi
    if dimension == 2:
        if fixed != "vertical":
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]], dtype=np.float32)

    elif dimension == 3:
        if fixed != "vertical":
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)

        else:
            phi = (np.random.rand() - 0.5) * np.pi

            # Create the first vector in carthesian coordinates
            u = np.array(
                [np.cos(theta) * np.cos(phi), np.sin(theta) * np.cos(phi), np.sin(phi)]
            )

            # Choose a random rotation angle
            alpha = np.random.rand() * 2 * np.pi

            # Create the rotation matrix with this vector and angle
            R = create_3D_rotations(np.reshape(u, (1, -1)), np.reshape(alpha, (1, -1)))[
                0
            ]

            R = R.astype(np.float32)

    # Add a small noise
    kernel_points = kernel_points + np.random.normal(
        scale=0.001, size=kernel_points.shape
    )

    # Scale kernels
    kernel_points = radius * kernel_points

    # Rotate kernels
    kernel_points = np.matmul(kernel_points, R)

    return kernel_points.astype(np.float32)
