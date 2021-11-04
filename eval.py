import argparse
import os
import sys
import pickle
import csv
sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m_multimodal import DatasetH36M
from motion_pred.utils.dataset_humaneva_multimodal import DatasetHumanEva
from motion_pred.utils.visualization import render_animation, render_animation_valcheck
from models.motion_pred_ours import *
from scipy.spatial.distance import pdist, squareform
from models import LinNF

from utils import util


def relative2absolute(x, parents, invert=False, x0=None):
    """
    x: [bs,..., jn, 3] or [bs,..., jn-1, 3] if invert
    x0: [1,..., jn, 3]
    parents: [-1,0,1 ...]
    """
    if not invert:
        xt = x[..., 1:, :] - x[..., parents[1:], :]
        xt = xt / torch.norm(xt, dim=-1, keepdim=True)
        return xt
    else:
        jn = x0.shape[-2]
        limb_l = torch.norm(x0[..., 1:, :] - x0[..., parents[1:], :], dim=-1, keepdim=True)
        xt = x * limb_l
        xt0 = torch.zeros_like(xt[..., :1, :])
        xt = torch.cat([xt0, xt], dim=-2)
        for i in range(1, jn):
            xt[..., i, :] = xt[..., parents[i], :] + xt[..., i, :]
        return xt


def denomarlize(*data):
    out = []
    for x in data:
        x = x * dataset.std + dataset.mean
        out.append(x)
    return out


def get_prediction(data, algo, sample_num, num_seeds=1, concat_hist=True, z=None):
    dct_m, idct_m = util.get_dct_matrix(t_pred + t_his)
    dct_m_all = dct_m.float().to(device)
    idct_m_all = idct_m.float().to(device)
    parts = cfg.nf_specs['parts']
    n_parts = len(parts)
    idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
    traj_np = data[..., 1:, :].transpose([0, 2, 3, 1])  # .reshape(traj_np.shape[0], traj_np.shape[1], -1)
    traj = tensor(traj_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
    bs, nj, _, _ = traj.shape
    inp = traj.reshape([bs, -1, traj.shape[-1]]).transpose(1, 2)
    inp = torch.matmul(dct_m_all[:cfg.n_pre], inp[:, idx_pad, :]).transpose(1, 2). \
        reshape([bs, nj, 3, -1]).reshape([bs, nj, -1])
    inp = inp.unsqueeze(1).repeat([1, cfg.nk, 1, 1]).reshape([bs * cfg.nk, nj, -1])

    # # sample diverse z
    # z = torch.randn([1, 3, cfg.nf_specs['nz']], dtype=dtype, device=device)
    # threshold = 30
    # max_search_step = 1000
    # for kk in range(sample_num * num_seeds - 1):
    #     zt = torch.randn([max_search_step, 3, cfg.nf_specs['nz']], dtype=dtype, device=device)
    #     dist = torch.norm(zt[:, None, :, :] - z[None, :, :, :], dim=-1).mean(dim=[1, 2])
    #     zt = zt[dist == torch.max(dist)][0]
    #     z = torch.cat([zt[None, :, :], z], dim=0)
    # # z = z.reshape([sample_num * num_seeds, X.shape[1], -1])
    # zz = torch.randn([sample_num * num_seeds, 2, cfg.nf_specs['nz']], dtype=dtype, device=device)
    # z = torch.cat([zz, z], dim=1)
    if algo == 'gcn':
        z = torch.randn([sample_num * num_seeds, n_parts, cfg.nf_specs['nz']], dtype=dtype, device=device)
        if args.fixlower:
            z[:, 0] = z[:1, 0]
        # z[:, :1] = z[:1, :1]
        Y = models['gcn'](inp, z)
        Y = Y.reshape([Y.shape[0], Y.shape[1], 3, cfg.n_pre]).reshape(
            [Y.shape[0], Y.shape[1] * 3, cfg.n_pre]).transpose(1, 2)
        Y = torch.matmul(idct_m_all[:, :cfg.n_pre], Y[:, :cfg.n_pre]).transpose(1, 2)[:, :, t_his:]
        X = traj[..., :t_his].reshape([traj.shape[0], traj.shape[1] * 3, t_his]).repeat([sample_num * num_seeds, 1, 1])
        # X = X.reshape([X.shape[0], X.shape[1], 3, n_his]).reshape([X.shape[0], X.shape[1] * 3, n_his]).transpose(1, 2)
        # X = torch.matmul(idct_m_his[:, :n_his], X).transpose(1, 2)

    # # aligh limb length
    # Y = Y.permute(0, 2, 1).reshape([cfg.nk, t_pred, 16, 3])
    # yt = torch.zeros([cfg.nk, t_pred, 17, 3], dtype=dtype, device=device)
    # yt[:, :, 1:] = Y
    # parents = dataset.skeleton.parents()
    # yt = relative2absolute(yt, parents)
    # x0 = torch.tensor(data[:, t_his:], dtype=dtype, device=device)
    # x0[:, :, 0] = 0
    # Y = relative2absolute(yt, parents=parents, invert=True, x0=x0)[:, :, 1:]
    # Y = Y.reshape([cfg.nk, t_pred, -1]).transpose(1, 2)

    if concat_hist:
        Y = torch.cat((X, Y), dim=-1)
    Y = Y.permute(0, 2, 1).contiguous().cpu().numpy()

    # n = 5
    # yt = tensor(Y, dtype=dtype, device=device)
    # yt = (yt - data_mean) / data_std
    # z, log_det_jacobian, _, _, _, _ = pose_prior(yt)
    # prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
    #                                    torch.tensor(1, dtype=dtype, device=device))
    # prior_lkh = prior.log_prob(z).sum(dim=2)
    # prior_logdetjac = log_det_jacobian.sum(dim=2)

    # z = torch.randn([sample_num * num_seeds, 2, 48], dtype=dtype, device=device).reshape(
    #     [sample_num * num_seeds, 2, -1])
    # zs = z[:, 0:1]
    # ze = z[:, -1:]
    # nf = t_pred + t_his
    # dz = (ze - zs) / (nf - 1)
    # zz = []
    # for i in range(nf):
    #     zz.append(zs + dz * i)
    # zz = torch.cat(zz, dim=1)
    # Y = pose_prior.inverse(zz)
    # Y = Y.transpose(1, 2)  # .reshape([Y.shape[0], 16, 3, -1])
    # Y = Y.permute(0, 2, 1).contiguous() * data_std + data_mean
    # Y = Y.cpu().numpy()

    if Y.shape[0] > 1:
        Y = Y.reshape(-1, sample_num, Y.shape[-2], Y.shape[-1])
    else:
        Y = Y[None, ...]
    return Y


def visualize():
    def post_process(pred, data):
        pred = pred.reshape(pred.shape[0], pred.shape[1], -1, 3)
        if cfg.normalize_data:
            pred = denomarlize(pred)
        pred = np.concatenate((np.tile(data[..., :1, :], (pred.shape[0], 1, 1, 1)), pred), axis=2)
        pred[..., :1, :] = 0
        return pred

    def pose_generator():

        while True:
            # while True:
            #     data, data_multimodal = dataset.sample(n_modality=10)
            #     # data = dataset.sample()
            #     dsum = np.sum(np.abs(data_multimodal), axis=(1, 2, 3))
            #     if len(np.where(dsum > 0)[0]) == 0:
            #         break

            data, data_multimodal = dataset.sample(n_modality=10)
            # data_multimodal[:, :, 0] = 0
            # gt
            gt = data[0].copy()
            gt[:, :1, :] = 0

            poses = {'context': gt, 'gt': gt}
            prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
                                               torch.tensor(1, dtype=dtype, device=device))
            for algo in vis_algos:
                pred = get_prediction(data, algo, nk, z=None)[0]

                # diversity and p(z) for gt
                div = compute_diversity(pred[:, t_his:])
                if 'gt' in poses.keys():
                    # get prior value
                    traj_tmp = tensor(gt[t_his:], dtype=dtype, device=device)
                    traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
                        [-1, dataset.traj_dim])
                    z, _ = pose_prior(traj_tmp)
                    prior_lkh = -prior.log_prob(z).sum(dim=1).mean().cpu().data.numpy()
                    poses[f'gt_{div:.1f}_p(z){prior_lkh:.1f}'] = gt
                    del poses['gt']

                # get prior value
                traj_tmp = tensor(pred[:, t_his:], dtype=dtype, device=device).reshape([-1, dataset.traj_dim])
                traj_tmp = traj_tmp.reshape([-1, dataset.traj_dim // 3, 3])
                tmp = torch.zeros_like(traj_tmp[:, :1, :])
                traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
                traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
                    [-1, dataset.traj_dim])
                z, _ = pose_prior(traj_tmp)
                prior_lkh = -prior.log_prob(z).sum(dim=1).reshape([-1, t_pred]).mean(dim=1).cpu().data.numpy()
                # prior_logdetjac = log_det_jacobian.sum(dim=2).mean(dim=1).cpu().data.numpy()

                pred = post_process(pred, data)
                for i in range(pred.shape[0]):
                    poses[f'{algo}_{i}_p(z){prior_lkh[i]:.1f}'] = pred[i]
                    # poses[f'{algo}_{i}'] = pred[i]

            yield poses

    pose_gen = pose_generator()
    # render_animation_valcheck(dataset.skeleton, pose_gen, vis_algos, cfg.t_his, ncol=12, output='out/video.mp4',
    #                           dataset=cfg.dataset)

    render_animation(dataset.skeleton, pose_gen, vis_algos, cfg.t_his, ncol=12, output='out/video.mp4')


def get_gt(data):
    gt = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
    return gt[:, t_his:, :]


"""metrics"""


def compute_diversity(pred, *args):
    if pred.shape[0] == 1:
        return 0.0
    dist = pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist.mean().item()
    return diversity


def compute_ade(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2).mean(axis=1)
    return dist.min()


def compute_fde(pred, gt, *args):
    diff = pred - gt
    dist = np.linalg.norm(diff, axis=2)[:, -1]
    return dist.min()


def compute_mmade(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_ade(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_mmfde(pred, gt, gt_multi):
    gt_dist = []
    for gt_multi_i in gt_multi:
        dist = compute_fde(pred, gt_multi_i)
        gt_dist.append(dist)
    gt_dist = np.array(gt_dist).mean()
    return gt_dist


def compute_pz(pred, *args):
    prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
                                       torch.tensor(1, dtype=dtype, device=device))
    # get prior value
    traj_tmp = tensor(pred, dtype=dtype, device=device)  # .reshape([-1, dataset.traj_dim])
    traj_tmp = traj_tmp.reshape([-1, dataset.traj_dim // 3, 3])
    tmp = torch.zeros_like(traj_tmp[:, :1, :])
    traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
    traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
        [-1, dataset.traj_dim])
    z, _ = pose_prior(traj_tmp)
    prior_lkh = -prior.log_prob(z).sum(dim=1).mean().cpu().data.numpy()
    return prior_lkh


def compute_stats():
    stats_func = {'Diversity': compute_diversity, 'ADE': compute_ade,
                  'FDE': compute_fde, 'MMADE': compute_mmade, 'MMFDE': compute_mmfde, 'NLL': compute_pz}
    stats_names = list(stats_func.keys())
    stats_meter = {x: {y: AverageMeter() for y in algos} for x in stats_names}

    data_gen = dataset.iter_generator(step=cfg.t_his)
    num_samples = 0
    num_seeds = args.num_seeds
    iv = 0
    for i, (data, _) in enumerate(data_gen):
        num_samples += 1
        gt = get_gt(data)
        gt_multi = traj_gt_arr[i]
        if gt_multi.shape[0] == 1:
            continue
        for algo in algos:
            pred = get_prediction(data, algo, sample_num=cfg.nk, num_seeds=num_seeds, concat_hist=False)
            for stats in stats_names:
                val = 0
                for pred_i in pred:
                    val += stats_func[stats](pred_i, gt, gt_multi) / num_seeds
                # if val > 50 and stats == 'Diversity':
                #     iv += 1
                #     break
                stats_meter[stats][algo].update(val)
        print('-' * 80)
        for stats in stats_names:
            str_stats = f'{num_samples:04d} {stats}: ' + ' '.join(
                [f'{x}: {y.val:.4f}({y.avg:.4f})' for x, y in stats_meter[stats].items()])
            print(str_stats)
        # break
    print(f'invalid samples {iv}, rate {iv / (nk * (i + 1))}')
    logger.info('=' * 80)
    surfix = f'#epo{args.iter_gcn}_fixlower_{args.fixlower}_nk{nk}_th{args.multimodal_threshold:.2f}'
    logger.info(surfix)
    for stats in stats_names:
        str_stats = f'Total {stats}: ' + ' '.join([f'{x}: {y.avg:.4f}' for x, y in stats_meter[stats].items()])
        logger.info(str_stats)
    logger.info('=' * 80)

    # with open('%s/stats_%s.csv' % (cfg.result_dir, args.num_seeds), 'w') as csv_file:
    #     writer = csv.DictWriter(csv_file, fieldnames=['Metric'] + algos)
    #     writer.writeheader()
    #     for stats, meter in stats_meter.items():
    #         new_meter = {x: y.avg for x, y in meter.items()}
    #         new_meter['Metric'] = stats
    #         writer.writerow(new_meter)

    whead = False
    if not os.path.exists('%s/stats_%s.csv' % (cfg.result_dir, args.num_seeds)):
        whead = True
    with open('%s/stats_%s.csv' % (cfg.result_dir, args.num_seeds), 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=['method'] + list(stats_meter.keys()))
        if whead:
            writer.writeheader()
        dict = {}
        for stats, meter in stats_meter.items():
            # print(stats)
            for x, y in meter.items():
                # print(x, y.avg)
                na = f'{x}_{surfix}'
                if na not in dict.keys():
                    dict[na] = {}
                dict[na][stats] = y.avg

        for stats, values in dict.items():
            new_meter = {x: y for x, y in values.items()}
            new_meter['method'] = stats
            writer.writerow(new_meter)


def get_multimodal_gt():
    all_data = []
    data_gen = dataset.iter_generator(step=cfg.t_his)
    for data, _ in data_gen:
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    pd = squareform(pdist(all_start_pose))
    traj_gt_arr = []
    num_mult = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data[ind][:, t_his:, :])
        num_mult.append(len(ind[0]))
    # np.savez_compressed('./data/data_3d_h36m_test.npz',data=all_data)
    # np.savez_compressed('./data/data_3d_humaneva15_test.npz',data=all_data)
    num_mult = np.array(num_mult)
    logger.info('')
    logger.info('')
    logger.info('=' * 80)
    logger.info(f'#1 future: {len(np.where(num_mult == 1)[0])}/{pd.shape[0]}')
    logger.info(f'#<10 future: {len(np.where(num_mult < 10)[0])}/{pd.shape[0]}')
    return traj_gt_arr


def get_multimodal_gt2():
    all_data = []
    data_gen = dataset.iter_generator(step=cfg.t_his)
    for data in data_gen:
        data = data[..., 1:, :].reshape(data.shape[0], data.shape[1], -1)
        all_data.append(data)
    all_data = np.concatenate(all_data, axis=0)
    all_data2 = np.concatenate(
        (all_data, dataset.data_candi['S9'][:, :, 1:].reshape([-1, t_pred + t_his, dataset.traj_dim])), axis=0)
    all_start_pose = all_data[:, t_his - 1, :]
    all_start_pose2 = all_data2[:, t_his - 1, :]
    # pd = np.linalg.norm(all_start_pose[:, None, :] - all_start_pose2[None, :, :], axis=2)
    pd = squareform(pdist(all_start_pose2))
    pd = pd[:all_data.shape[0]]
    traj_gt_arr = []
    num_mult = []
    for i in range(pd.shape[0]):
        ind = np.nonzero(pd[i] < args.multimodal_threshold)
        traj_gt_arr.append(all_data2[ind][:, t_his:, :])
        num_mult.append(len(ind[0]))
    num_mult = np.array(num_mult)
    return traj_gt_arr


if __name__ == '__main__':

    all_algos = ['gcn']
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        default='humaneva')
    parser.add_argument('--mode', default='stats')
    parser.add_argument('--data', default='test')
    parser.add_argument('--action', default='all')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--multimodal_threshold', type=float, default=0.5)
    parser.add_argument('--multimodal_th_high', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=1)
    parser.add_argument('--n_pre', type=int, default=10)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--nk', type=int, default=-1)
    parser.add_argument('--fixlower', action='store_true', default=False)
    parser.add_argument('--num_coupling_layer', type=int, default=4)
    for algo in all_algos:
        parser.add_argument('--iter_%s' % algo, type=int, default=500)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if args.gpu_index >= 0 and \
                                                           torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    torch.set_grad_enabled(False)
    cfg = Config(args.cfg)
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    algos = []
    for algo in all_algos:
        # iter_algo = 'iter_%s' % algo
        # num_algo = 'num_vae_epoch'  # % algo
        # setattr(args, iter_algo, getattr(cfg, num_algo))
        algos.append(algo)
    vis_algos = algos.copy()

    if args.action != 'all':
        args.action = set(args.action.split(','))

    """parameter"""
    if args.mode == 'vis':
        cfg.nk = 10
    else:
        if args.nk > 0:
            cfg.nk = args.nk
        else:
            cfg.nk = 50
    nz = cfg.nz
    nk = cfg.nk
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    n_his = args.n_his
    cfg.n_his = n_his
    # n_pre = args.n_pre
    if 'n_pre' not in cfg.nf_specs.keys():
        n_pre = args.n_pre
    else:
        n_pre = cfg.nf_specs['n_pre']
    cfg.n_pre = n_pre
    cfg.num_coupling_layer = args.num_coupling_layer

    """data"""
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls(args.data, t_his, t_pred, actions=args.action, use_vel=cfg.use_vel,
                          multimodal_path=cfg.nf_specs[
                              'multimodal_path'] if 'multimodal_path' in cfg.nf_specs.keys() else None,
                          data_candi_path=cfg.nf_specs[
                              'data_candi_path'] if 'data_candi_path' in cfg.nf_specs.keys() else None)
    if args.data == 'test':
        traj_gt_arr = get_multimodal_gt()

    """models"""
    model_generator = {
        'gcn': get_model
        # ,
        # 'dlow': get_dlow_model,

    }
    models = {}
    for algo in algos:
        models[algo], pose_prior = model_generator[algo](cfg, dataset.traj_dim // 3, args.cfg)
        models[algo].float()
        model_path = getattr(cfg, f"vae_model_path") % getattr(args, f'iter_{algo}')
        print(f'loading {algo} model from checkpoint: {model_path}')
        model_cp = pickle.load(open(model_path, "rb"))
        models[algo].load_state_dict(model_cp['model_dict'])
        models[algo].to(device)
        models[algo].eval()

        LinNF.LinNF(data_dim=dataset.traj_dim, num_layer=3)
        cp_path = './results_pretrained/h36m_linNF_pose_prior_float/models/vae_0025.p' if cfg.dataset == 'h36m' else \
            './results_pretrained/humaneva_nf/models/vae_0025.p'
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        pose_prior.load_state_dict(model_cp['model_dict'])
        pose_prior.to(device)
        pose_prior.eval()

    if cfg.normalize_data:
        dataset.normalize_data(model_cp['meta']['mean'], model_cp['meta']['std'])

    if args.mode == 'vis':
        visualize()
    elif args.mode == 'stats':
        compute_stats()
