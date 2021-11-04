# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pickle
from vpython import *
import time

from utils import valid_angle_check


def render_animation(skeleton, poses_generator, algos, t_hist, fix_0=True, azim=0.0, output=None, size=6, ncol=5,
                     bitrate=3000):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """

    all_poses = next(poses_generator)
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    t_total = next(iter(all_poses.values())).shape[0]
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                        all_poses.items()))
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(nrow, ncol, index + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        ax.set_title(title, y=1.2)
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.4, hspace=0)
    poses = list(poses.values())

    anim = None
    initialized = False
    animating = True
    find = 0
    hist_lcol, hist_rcol = 'black', 'red'
    pred_lcol, pred_rcol = 'purple', 'green'

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized
        if i < t_hist:
            lcol, rcol = hist_lcol, hist_rcol
        else:
            lcol, rcol = pred_lcol, pred_rcol

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2 + trajectories[n][i, 2]])
            # ax.plot([0, 0.1],
            #         [0, 0],
            #         [0, 0], c='r')
            # ax.plot([0, 0],
            #         [0, 0.1],
            #         [0, 0], c='g')
            # ax.plot([0, 0],
            #         [0, 0],
            #         [0, 0.1], c='b')
        if not initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                    # if n == 0:
                    # if j == 1:
                    #     for tx in ax.texts:
                    #         tx.remove()
                    #     for tx in ax.texts:
                    #         tx.remove()
                    # ax.text(pos[j, 0], pos[j, 1], pos[j, 2], f'{j}', None)
            initialized = True
        else:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
                    lines_3d[n][j - 1][0].set_color(col)
                    # if j == 1:
                    #     for tx in ax.texts:
                    #         tx.remove()
                    #     for tx in ax.texts:
                    #         tx.remove()
                    # ax.text(pos[j, 0], pos[j, 1], pos[j, 2], f'{j}', None)

    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                            all_poses.items()))
        for ax, title in zip(ax_3d, poses.keys()):
            ax.set_title(title, y=1.2)
        poses = list(poses.values())

    def save_figs():
        nonlocal algo, find
        old_algo = algo
        for algo in algos:
            reload_poses()
            update_video(t_total - 1)
            fig.savefig('out/%d_%s.png' % (find, algo), dpi=400, transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 30
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)
    show_animation()
    plt.show()


def render_animation_withsaveimgs(skeleton, poses_generator, algos, t_hist, fix_0=True, azim=0.0, output=None, size=6,
                                  ncol=5,
                                  bitrate=3000, hist_col=('#3498db', '#e74c3c'), pred_col=('#8a11ed', '#3ce79a')):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """

    all_poses = next(poses_generator)
    IDX = 0
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    t_total = next(iter(all_poses.values())).shape[0]
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                        all_poses.items()))
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(nrow, ncol, index + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0
        ax.set_title(title, y=1.0)
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])
    fig.tight_layout()
    fig.suptitle(f"{IDX:03d}")
    fig.subplots_adjust(wspace=-0.4, hspace=0)
    poses = list(poses.values())

    anim = None
    initialized = False
    animating = True
    find = 0
    hist_lcol, hist_rcol = hist_col
    pred_lcol, pred_rcol = pred_col

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized, IDX, fig
        if i < t_hist:
            lcol, rcol = hist_lcol, hist_rcol
        else:
            lcol, rcol = pred_lcol, pred_rcol

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2 + trajectories[n][i, 2]])
            # ax.plot([0, 0.1],
            #         [0, 0],
            #         [0, 0], c='r')
            # ax.plot([0, 0],
            #         [0, 0.1],
            #         [0, 0], c='g')
            # ax.plot([0, 0],
            #         [0, 0],
            #         [0, 0.1], c='b')
        if not initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
                    # if n == 0:
                    #     ax.text(pos[j, 0], pos[j, 1], pos[j, 2], f'{j}', None)
            initialized = True
        else:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
                    lines_3d[n][j - 1][0].set_color(col)

    def show_animation():
        nonlocal anim, IDX
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or algo == x[0].split('_')[0] or x[0].startswith('gt'),
                            all_poses.items()))
        for ax, title in zip(ax_3d, poses.keys()):
            ax.set_title(title, y=1.0)
        poses = list(poses.values())

    def save_figs():
        nonlocal algo, find
        old_algo = algo
        for algo in algos:
            reload_poses()
            update_video(t_total - 1)
            fig.savefig('out/%d_%s.png' % (find, algo), dpi=400, transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim, IDX

        if event.key == 'd':
            all_poses = next(poses_generator)
            IDX += 1
            fig.suptitle(f"{IDX:03d}")
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == 'e':
            save_img()
        elif event.key == 'r':
            print('save 3d renders')
            save_3d_renders()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 30
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    def save_img():
        # import matplotlib.pyplot as plt
        # skeleton = dataset.skeleton
        parents = skeleton.parents()
        # rcol, lcol = ("#3498db", "#e74c3c")
        # hist_lcol, hist_rcol = 'black', 'red'
        # pred_lcol, pred_rcol = 'purple', 'green'
        keys = all_poses.keys()
        os.makedirs(os.path.dirname(output), exist_ok=True)
        np.savez_compressed(f'{output}/{IDX:03d}.npz', all_poses=all_poses, skeleton=skeleton)
        for k in keys:
            if 'vae' in k or 'context' in k:
                continue
            figtmp = plt.figure(figsize=(30, 3))
            plt.axis('equal')
            plt.axis('off')
            # project
            tx = 90 / 180 * np.pi
            cxs = np.cos(tx)
            sxs = np.sin(tx)
            rx = np.array([[1, 0, 0], [0, cxs, -sxs], [0, sxs, cxs]])
            ry = np.array([[cxs, 0, -sxs], [0, 1, 0], [sxs, 0, cxs]])
            pose = all_poses[k]
            pose = np.matmul(np.matmul(pose.reshape([-1, 3]), rx), ry).reshape(pose.shape)
            pose[..., :, 2] = pose[..., :, 2] + 4
            p2d = pose[..., :2] / pose[..., 2:]
            skip_rate = 6
            lw = 3
            p2d = np.concatenate((p2d[0:25:skip_rate // 2], p2d[25::skip_rate]), axis=0)
            n = p2d.shape[0]
            # step = ((p2d[:, :, 0]).max(axis=1) - (p2d[:, :, 0]).min(axis=1)).max()
            step = 0.16
            step = (np.arange(n) * step)[:, None, None]
            p2d[:, :, :1] = p2d[:, :, :1] + step
            for i in range(p2d.shape[0]):
                # plt.cla()
                pos = p2d[i]
                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue
                    if i < 5:
                        col = hist_rcol if j in skeleton.joints_right() else hist_lcol
                    else:
                        col = pred_rcol if j in skeleton.joints_right() else pred_lcol
                    plt.plot([pos[j, 0], pos[j_parent, 0]], [pos[j, 1], pos[j_parent, 1]], c=col, linewidth=lw)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            plt.savefig(f'{output}/{IDX:03d}_{k}.pdf')
            plt.close(figtmp)

    def save_img2():
        # import matplotlib.pyplot as plt
        # skeleton = dataset.skeleton
        parents = skeleton.parents()
        # rcol, lcol = ("#3498db", "#e74c3c")
        # hist_lcol, hist_rcol = 'black', 'red'
        # pred_lcol, pred_rcol = 'purple', 'green'
        keys = all_poses.keys()
        os.makedirs(os.path.dirname(output), exist_ok=True)
        np.savez_compressed(f'{output}/{IDX:03d}.npz', all_poses=all_poses, skeleton=skeleton)
        for k in keys:
            if 'vae' in k or 'context' in k:
                continue
            figtmp = plt.figure(figsize=(30, 3))
            plt.axis('equal')
            plt.axis('off')
            # project
            tx = 90 / 180 * np.pi
            cxs = np.cos(tx)
            sxs = np.sin(tx)
            rx = np.array([[1, 0, 0], [0, cxs, -sxs], [0, sxs, cxs]])
            ry = np.array([[cxs, 0, -sxs], [0, 1, 0], [sxs, 0, cxs]])
            pose = all_poses[k]
            pose = np.matmul(np.matmul(pose.reshape([-1, 3]), rx), ry).reshape(pose.shape)
            pose[..., :, 2] = pose[..., :, 2] + 4
            p2d = pose[..., :2] / pose[..., 2:]
            skip_rate = 10
            lw = 3
            p2d = np.concatenate((p2d[0:25:skip_rate // 2], p2d[25::skip_rate]), axis=0)
            n = p2d.shape[0]
            # step = ((p2d[:, :, 0]).max(axis=1) - (p2d[:, :, 0]).min(axis=1)).max()
            step = 0.16
            step = (np.arange(n) * step)[:, None, None]
            p2d[:, :, :1] = p2d[:, :, :1] + step
            for i in range(p2d.shape[0]):
                # plt.cla()
                pos = p2d[i]
                for j, j_parent in enumerate(parents):
                    if j_parent == -1:
                        continue
                    if i < 5:
                        col = hist_rcol if j in skeleton.joints_right() else hist_lcol
                    else:
                        col = pred_rcol if j in skeleton.joints_right() else pred_lcol
                    plt.plot([pos[j, 0], pos[j_parent, 0]], [pos[j, 1], pos[j_parent, 1]], c=col, linewidth=lw)
            plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
            plt.savefig(f'{output}/{IDX:03d}_{k}.pdf')
            plt.close(figtmp)

    def save_3d_renders():
        # blue = vector(0.5, 0.5, 1)
        # purple = vector(0.8, 0.3, 0.8)
        # red = vector(1, 0.5, 0.5)
        # green = vector(0.5, 1, 0.5)
        blue = vector(0.203125, 0.59375, 0.85546875)
        red = vector(0.90234375, 0.296875, 0.234375)
        purple = vector(0.5390625, 0.06640625, 0.92578125)
        green = vector(0.234375, 0.90234375, 0.6015625)
        gray = vector(0.5, 0.5, 0.5)

        hist_col = (blue, red)
        pred_col = (purple, green)
        hist_lcol, hist_rcol = hist_col
        pred_lcol, pred_rcol = pred_col
        skip_rate = 6
        src = 'h36m_nsamp50'
        ind_samp = '001'
        ind_seq = 'dlow_7'
        cyl_size = 0.06
        # all_poses = np.load(f'./results_pretrained/{src}/results_pretrained/vis/{ind_samp}.npz', allow_pickle=True)['all_poses'].item()
        # skeleton = np.load(f'./results_pretrained/{src}/results_pretrained/vis/{ind_samp}.npz', allow_pickle=True)['skeleton'].item()
        # parents = skeleton.parents()
        parts = [[6, 7, 8, 9], [6, 7, 8, 9, 0, 1, 2], [6, 7, 8, 9, 0, 1, 2, 3, 4, 5],
                 [6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 13, 14, 15], [10, 11, 12]]
        for k in all_poses.keys():
            if 'vae' in k or 'context' in k:
                continue
            if 'dlow_3' not in k:
                continue
            poses = all_poses[k]
            poses = np.concatenate((poses[0:25:2], poses[25::2]), axis=0)
            t_his = 13
            for n, pos in enumerate(poses):
                scene = canvas(width=600, height=900, background=vector(1, 1, 1))

                scene.camera.pos = vector(0, 1, 0.5)
                scene.camera.axis = vector(0, -2, -1)
                scene.up = vector(0, 0, 1)
                scene.lights[0].direction = vector(0, 0.44, 0.88)
                scene.lights[0].color = color.white
                scene.lights[1].direction = vector(0.22, -0.44, -0.88)

                scene.camera.pos = vector(-1, 0, 0.5)
                scene.camera.axis = vector(2, 0, -1)
                scene.up = vector(0, 0, 1)
                scene.lights[0].direction = vector(0, 0.44, 0.88)
                scene.lights[0].color = color.white
                scene.lights[1].direction = vector(0.22, -0.44, -0.88)

                # scene.camera.pos = vector(1, -0.5, 0.8)
                # scene.camera.axis = vector(-1, 0.5, -0.5)
                # scene.up = vector(0, 0, 1)
                # scene.lights[0].direction = vector(0.22, 0.44, 0.88)
                # scene.lights[0].color = color.gray(0.3)
                # scene.lights[1].direction = vector(0, -0.44, 0.88)
                # scene.lights[1].color = color.gray(0.8)

                for i, jpos in enumerate(pos):
                    if parents[i] < 0:
                        continue
                    if n < t_his:
                        lcol, rcol = hist_lcol, hist_rcol
                    else:
                        lcol, rcol = pred_lcol, pred_rcol

                    col = rcol if i in skeleton.joints_right() else lcol
                    # if (i - 1) in parts[3]:
                    #     col = gray
                    direct = pos[parents[i]] - jpos
                    leng = np.linalg.norm(direct)
                    direct = direct / leng
                    cylinder(pos=vector(jpos[0], jpos[1], jpos[2]), axis=vector(direct[0], direct[1], direct[2]),
                             size=vector(leng, cyl_size, cyl_size), color=col)
                scene.capture(f'{IDX}_{k}_{n}')
                time.sleep(0.8)
                scene.delete()

    fig.canvas.mpl_connect('key_press_event', on_key)
    show_animation()
    plt.show()


def render_animation_valcheck(skeleton, poses_generator, algos, t_hist, fix_0=True, azim=0.0, output=None, size=6,
                              ncol=5,
                              bitrate=3000, dataset='h36m'):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """

    all_poses = next(poses_generator)
    algo = algos[0] if len(algos) > 0 else next(iter(all_poses.keys()))
    t_total = next(iter(all_poses.values())).shape[0]
    poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or x[0].startswith('gt') or algo == x[0].split('_')[0],
                        all_poses.items()))
    plt.ioff()
    nrow = int(np.ceil(len(poses) / ncol))
    fig = plt.figure(figsize=(size * ncol, size * nrow))
    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7

    valid_angle_cos = pickle.load(open('./data_stats_valid_angle.p', "rb")) if dataset == 'h36m' else pickle.load(
        open('./data/humaneva_valid_angle.p', "rb"))
    ang_names = valid_angle_cos.keys()

    for index, (title, data) in enumerate(poses.items()):
        ax = fig.add_subplot(nrow, ncol, index + 1, projection='3d')
        ax.view_init(elev=15., azim=azim)
        ax.set_xlim3d([-radius / 2, radius / 2])
        ax.set_zlim3d([0, radius])
        ax.set_ylim3d([-radius / 2, radius / 2])
        # ax.set_aspect('equal')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.dist = 5.0

        # check valid angle
        # angle_cos = h36m_valid_angle_check(poses[title][:, 1:])
        angle_cos = valid_angle_check.h36m_valid_angle_check(
            poses[title][:, 1:]) if dataset == 'h36m' else valid_angle_check.humaneva_valid_angle_check(
            poses[title][:, 1:])
        for an in ang_names:
            an_cos = angle_cos[an]
            if len(an_cos) > 0:
                c1 = an_cos < (valid_angle_cos[an][0] if valid_angle_cos[an][0] >= -0.95 else -1.0005)
                c2 = an_cos > (valid_angle_cos[an][1] if valid_angle_cos[an][1] <= 0.95 else 1.0005)
                if np.any(np.logical_or(c1, c2)):
                    title = title + f'\n{an}'
        ax.set_title(title, y=1.0, fontsize=8)

        # ax.set_title(title, y=1.2)
        ax.set_axis_off()
        ax.patch.set_alpha(0.0)
        ax_3d.append(ax)
        lines_3d.append([])
        trajectories.append(data[:, 0, [0, 1]])

    fig.tight_layout()
    fig.subplots_adjust(wspace=-0.4, hspace=0)
    poses = list(poses.values())

    anim = None
    initialized = False
    animating = True
    find = 0
    hist_lcol, hist_rcol = 'black', 'red'
    pred_lcol, pred_rcol = 'purple', 'green'

    parents = skeleton.parents()

    def update_video(i):
        nonlocal initialized
        if i < t_hist:
            lcol, rcol = hist_lcol, hist_rcol
        else:
            lcol, rcol = pred_lcol, pred_rcol

        for n, ax in enumerate(ax_3d):
            if fix_0 and n == 0 and i >= t_hist:
                continue
            trajectories[n] = poses[n][:, 0, [0, 1, 2]]
            ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius / 2 + trajectories[n][i, 2], radius / 2 + trajectories[n][i, 2]])
            # ax.plot([0, 0.1],
            #         [0, 0],
            #         [0, 0], c='r')
            # ax.plot([0, 0],
            #         [0, 0.1],
            #         [0, 0], c='g')
            # ax.plot([0, 0],
            #         [0, 0],
            #         [0, 0.1], c='b')
        if not initialized:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
            initialized = True
        else:

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue

                col = rcol if j in skeleton.joints_right() else lcol
                for n, ax in enumerate(ax_3d):
                    if fix_0 and n == 0 and i >= t_hist:
                        continue
                    pos = poses[n][i]
                    lines_3d[n][j - 1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j - 1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j - 1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')
                    lines_3d[n][j - 1][0].set_color(col)

    def show_animation():
        nonlocal anim
        if anim is not None:
            anim.event_source.stop()
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=0, repeat=True)
        plt.draw()

    def reload_poses():
        nonlocal poses
        poses = dict(filter(lambda x: x[0] in {'gt', 'context'} or x[0].startswith('gt') or algo == x[0].split('_')[0],
                            all_poses.items()))

        valid_angle_cos = pickle.load(open('./data_stats_valid_angle.p', "rb")) if dataset == 'h36m' else pickle.load(
            open('./data/humaneva_valid_angle.p', "rb"))
        ang_names = valid_angle_cos.keys()
        for ax, title in zip(ax_3d, poses.keys()):
            # print(dataset)
            # check valid angle
            angle_cos = valid_angle_check.h36m_valid_angle_check(
                poses[title][:, 1:]) if dataset == 'h36m' else valid_angle_check.humaneva_valid_angle_check(
                poses[title][:, 1:])
            for an in ang_names:
                an_cos = angle_cos[an]
                if len(an_cos) > 0:
                    c1 = an_cos < (valid_angle_cos[an][0] if valid_angle_cos[an][0] >= -0.95 else -1.0005)
                    c2 = an_cos > (valid_angle_cos[an][1] if valid_angle_cos[an][1] <= 0.95 else 1.0005)
                    if np.any(np.logical_or(c1, c2)):
                        title = title + f'\n{an}'
            ax.set_title(title, y=1.0, fontsize=8)
        poses = list(poses.values())

    def save_figs():
        nonlocal algo, find
        old_algo = algo
        for algo in algos:
            reload_poses()
            update_video(t_total - 1)
            fig.savefig('out/%d_%s.png' % (find, algo), dpi=400, transparent=True)
        algo = old_algo
        find += 1

    def on_key(event):
        nonlocal algo, all_poses, animating, anim

        if event.key == 'd':
            all_poses = next(poses_generator)
            reload_poses()
            show_animation()
        elif event.key == 'c':
            save()
        elif event.key == ' ':
            if animating:
                anim.event_source.stop()
            else:
                anim.event_source.start()
            animating = not animating
        elif event.key == 'v':  # save images
            if anim is not None:
                anim.event_source.stop()
                anim = None
            save_figs()
        elif event.key.isdigit():
            algo = algos[int(event.key) - 1]
            reload_poses()
            show_animation()

    def save():
        nonlocal anim

        fps = 30
        anim = FuncAnimation(fig, update_video, frames=np.arange(0, poses[0].shape[0]), interval=1000 / fps,
                             repeat=False)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if output.endswith('.mp4'):
            Writer = writers['ffmpeg']
            writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
            anim.save(output, writer=writer)
        elif output.endswith('.gif'):
            anim.save(output, dpi=80, writer='imagemagick')
        else:
            raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
        print(f'video saved to {output}!')

    fig.canvas.mpl_connect('key_press_event', on_key)
    show_animation()
    plt.show()
