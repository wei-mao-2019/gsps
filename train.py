import os
import sys
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_h36m_multimodal import DatasetH36M
from motion_pred.utils.dataset_humaneva_multimodal import DatasetHumanEva
from models.motion_pred_ours import *
from utils import util, valid_angle_check


def joint_loss(Y_g):
    parts = cfg.nf_specs['parts']
    parts_idx = [(np.array(p) * 3).tolist() + (np.array(p) * 3 + 1).tolist() + (np.array(p) * 3 + 2).tolist()
                 for p in parts]
    nparts = len(parts)
    if 'alphas' in cfg.nf_specs.keys():
        alpha = cfg.nf_specs['alphas'][0]
        beta = cfg.nf_specs['alphas'][1]
    else:
        alpha = 100
        beta = 300

    loss = []
    Y_g = Y_g.permute(1, 0, 2).contiguous()
    Y_g = Y_g.view([Y_g.shape[0] // cfg.nk ** nparts] + [cfg.nk] * nparts + [Y_g.shape[1], -1])

    assert nparts == 2
    mask = torch.tril(torch.ones([cfg.nk, cfg.nk], device=device)) == 0
    yt = Y_g[:, :, 0, ...][..., parts_idx[0]].reshape([Y_g.shape[0], cfg.nk, -1])
    # pdist = (yt[:, :, None] - yt[:, None, :]).abs()[:, mask]
    pdist = torch.cdist(yt, yt, p=1)[:, mask]
    loss.append((-pdist / alpha).exp().mean())
    yt = Y_g[..., parts_idx[1]].reshape([Y_g.shape[0] * cfg.nk, cfg.nk, -1])
    # pdist = (yt[:, :, None] - yt[:, None, :]).abs()[:, mask]
    pdist = torch.cdist(yt, yt, p=1)[:, mask]
    loss.append((-pdist / beta).exp().mean())
    with torch.no_grad():
        mask = torch.tril(torch.ones([cfg.nk ** nparts, cfg.nk ** nparts], device=device)) == 0
        yt = Y_g.reshape([Y_g.shape[0], cfg.nk ** nparts, -1])
        pdist = torch.cdist(yt, yt, p=2)[:, mask]
        # loss.append(pdist.mean())
    return loss, pdist.mean()


def recon_loss(Y_g, Y, Y_mm):
    parts = cfg.nf_specs['parts']
    nparts = len(parts)
    Y_g = Y_g.view(Y_g.shape[0], -1, cfg.nk ** nparts, Y_g.shape[2])
    diff = Y_g - Y.unsqueeze(2)
    dist = diff.pow(2).sum(dim=-1).sum(dim=0)
    loss_recon = dist.min(dim=1)[0].mean()
    with torch.no_grad():
        ade = torch.norm(diff, dim=-1).mean(dim=0).min(dim=1)[0].mean()

    diff = Y_g[:, :, :, None, :] - Y_mm[:, :, None, :, :]
    mask = Y_mm.abs().sum(-1).sum(0) > 1e-6
    dist = diff.pow(2).sum(dim=-1).sum(dim=0)
    loss_recon_multi = dist.min(dim=1)[0][mask].mean()
    if torch.isnan(loss_recon_multi):
        loss_recon_multi = torch.zeros_like(loss_recon)
    return loss_recon, loss_recon_multi, ade


def angle_loss(y):
    ang_names = list(valid_ang.keys())
    y = y.reshape([-1, y.shape[-1]])
    ang_cos = valid_angle_check.h36m_valid_angle_check_torch(
        y) if cfg.dataset == 'h36m' else valid_angle_check.humaneva_valid_angle_check_torch(y)
    loss = tensor(0, dtype=dtype, device=device)
    b = 1
    for an in ang_names:
        lower_bound = valid_ang[an][0]
        if lower_bound >= -0.98:
            # loss += torch.exp(-b * (ang_cos[an] - lower_bound)).mean()
            if torch.any(ang_cos[an] < lower_bound):
                # loss += b * torch.exp(-(ang_cos[an][ang_cos[an] < lower_bound] - lower_bound)).mean()
                loss += (ang_cos[an][ang_cos[an] < lower_bound] - lower_bound).pow(2).mean()
        upper_bound = valid_ang[an][1]
        if upper_bound <= 0.98:
            # loss += torch.exp(b * (ang_cos[an] - upper_bound)).mean()
            if torch.any(ang_cos[an] > upper_bound):
                # loss += b * torch.exp(ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).mean()
                loss += (ang_cos[an][ang_cos[an] > upper_bound] - upper_bound).pow(2).mean()
    return loss


def loss_function(traj_est, traj, traj_multimodal, prior_lkh, prior_logdetjac):
    lambdas = cfg.nf_specs['lambdas']
    parts = cfg.nf_specs['parts']
    nparts = len(parts)
    nj = dataset.traj_dim // 3

    # diversity loss
    Y_g = traj_est[t_his:]
    JL, div = joint_loss(Y_g)

    # reconstruction loss
    Y = traj[t_his:]
    Y_multimodal = traj_multimodal[t_his:]
    RECON, RECON_mm, ade = recon_loss(Y_g, Y, Y_multimodal)

    # recover history
    xest = traj_est[:t_his].reshape([t_his, cfg.batch_size, cfg.nk ** nparts, -1])
    xgt = traj[:t_his].unsqueeze(2)
    loss_x = torch.mean((xest - xgt).pow(2).sum(dim=-1))

    # maintain limb length
    parent = dataset.skeleton.parents()
    tmp = traj[0].reshape([cfg.batch_size, nj, 3])
    pgt = torch.zeros([cfg.batch_size, nj + 1, 3], dtype=dtype, device=device)
    pgt[:, 1:] = tmp
    limbgt = torch.norm(pgt[:, 1:] - pgt[:, parent[1:]], dim=2)[None, :, None, :]
    tmp = traj_est.reshape([-1, cfg.batch_size, cfg.nk ** nparts, nj, 3])
    pest = torch.zeros([tmp.shape[0], cfg.batch_size, cfg.nk ** nparts, nj + 1, 3], dtype=dtype, device=device)
    pest[:, :, :, 1:] = tmp
    limbest = torch.norm(pest[:, :, :, 1:] - pest[:, :, :, parent[1:]], dim=4)
    loss_limb = torch.mean((limbgt - limbest).pow(2).sum(dim=3))

    # angle loss
    loss_ang = angle_loss(Y_g)

    loss_r = loss_x * lambdas[0] + loss_limb * lambdas[1] \
             + JL[0] * lambdas[2] + JL[1] * lambdas[3] + RECON * lambdas[4] + RECON_mm * lambdas[5] \
             - prior_lkh.mean() * lambdas[6]  # - prior_logdetjac.mean() * lambdas[7]

    if loss_ang > 0:
        loss_r += loss_ang * lambdas[8]
    return loss_r, np.array([loss_r.item(), loss_x.item(), loss_limb.item(), loss_ang.item(),
                             JL[0].item(), JL[1].item(), div.item(), RECON.item(), RECON_mm.item(), ade.item(),
                             prior_lkh.mean().item(), prior_logdetjac.mean().item()])


def train(epoch):
    model.train()
    t_s = time.time()
    train_losses = 0
    train_grad = 0
    train_grad_d = 0
    total_num_sample = 0
    n_modality = 10
    loss_names = ['LOSS', 'loss_cont', 'loss_limb', 'loss_ang', 'loss_DIV_L', 'loss_DIV_U', 'DIV',
                  'RECON', 'RECON_multi', "ADE", 'p(z)', 'logdet']
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size,
                                           n_modality=n_modality)
    prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
                                       torch.tensor(1, dtype=dtype, device=device))
    # generator_d = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size)

    dct_m, idct_m = util.get_dct_matrix(t_pred + t_his)
    dct_m_all = dct_m.float().to(device)
    idct_m_all = idct_m.float().to(device)
    parts = cfg.nf_specs['parts']
    n_parts = len(parts)
    idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
    k = 1
    for traj_np, traj_multimodal_np in generator:
        with torch.no_grad():
            traj_np = traj_np[..., 1:, :].transpose([0, 2, 3, 1])  # .reshape(traj_np.shape[0], traj_np.shape[1], -1)
            traj = tensor(traj_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
            bs, nj, _, _ = traj.shape
            inp = traj.reshape([bs, -1, t_his + t_pred]).transpose(1, 2)
            inp = torch.matmul(dct_m_all[:cfg.n_pre], inp[:, idx_pad, :]).transpose(1, 2). \
                reshape([bs, nj, 3, -1]).reshape([bs, nj, -1])

            traj_multimodal_np = traj_multimodal_np[..., 1:, :]  # [bs, modality, seqn, jn, 3]
            traj_multimodal_np = traj_multimodal_np.reshape([bs, n_modality, t_his + t_pred, -1]).transpose(
                [2, 0, 1, 3])
            traj_multimodal = tensor(traj_multimodal_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()

            inp = inp.unsqueeze(1).repeat([1, (cfg.nk ** n_parts), 1, 1]).reshape(
                [bs * (cfg.nk ** n_parts), nj, -1])
            z = None
            for _ in range(n_parts):
                if z is None:
                    zt = torch.randn([bs, cfg.nk, 1, cfg.nf_specs['nz']], dtype=dtype, device=device)
                    z = zt
                else:
                    z = z.repeat_interleave(cfg.nk, dim=1)
                    zt = torch.randn([bs, z.shape[1], 1, cfg.nf_specs['nz']], dtype=dtype, device=device)
                    z = torch.cat([z, zt], dim=2)
            z = z.reshape([-1, n_parts, cfg.nf_specs['nz']])

        # train generator
        xt = model(inp, z)
        xt = xt.reshape([bs * (cfg.nk ** n_parts), nj, 3, -1]).reshape([bs * (cfg.nk ** n_parts), nj * 3, -1]) \
            .transpose(1, 2)
        traj_est = torch.matmul(idct_m_all[:, :cfg.n_pre], xt).transpose(0, 1)
        traj = traj.reshape([bs, -1, t_his + t_pred]).permute([2, 0, 1])

        # to save computation
        ran = np.random.uniform()
        if ran > 0.67:
            traj_tmp = traj_est[t_his::3].reshape([-1, traj_est.shape[-1] // 3, 3])
            tmp = torch.zeros_like(traj_tmp[:, :1, :])
            traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
            traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
                [-1, traj_est.shape[-1]])
        elif ran > 0.33:
            traj_tmp = traj_est[t_his + 1::3].reshape([-1, traj_est.shape[-1] // 3, 3])
            tmp = torch.zeros_like(traj_tmp[:, :1, :])
            traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
            traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
                [-1, traj_est.shape[-1]])
        else:
            traj_tmp = traj_est[t_his + 2::3].reshape([-1, traj_est.shape[-1] // 3, 3])
            tmp = torch.zeros_like(traj_tmp[:, :1, :])
            traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
            traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
                [-1, traj_est.shape[-1]])
        z, prior_logdetjac = pose_prior(traj_tmp)

        prior_lkh = prior.log_prob(z).sum(dim=-1)
        # prior_logdetjac = log_det_jacobian.sum(dim=2)

        loss, losses = loss_function(traj_est, traj, traj_multimodal, prior_lkh, prior_logdetjac)
        # if torch.isinf(loss):
        #     print(1)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=100)
        train_grad += grad_norm
        optimizer.step()
        train_losses += losses
        total_num_sample += 1
        # print(torch.cuda.memory_allocated()/1024/1024)
        del loss, z, inp, xt, traj_est
        # print(torch.cuda.memory_allocated())

    scheduler.step()
    # dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])

    # average cost of log time 20s
    tb_logger.add_scalar('train_grad', train_grad / total_num_sample, epoch)
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars(name, {'train': loss}, epoch)

    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, time.time() - t_s, losses_str, lr))


def val(epoch):
    model.eval()
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    n_modality = 10
    loss_names = ['LOSS', 'loss_cont', 'loss_limb', 'loss_ang', 'loss_DIV_L', 'loss_DIV_U', 'DIV',
                  'RECON', 'RECON_multi', "ADE", 'p(z)', 'logdet']
    generator = dataset_test.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size)
    prior = torch.distributions.Normal(torch.tensor(0, dtype=dtype, device=device),
                                       torch.tensor(1, dtype=dtype, device=device))

    with torch.no_grad():
        dct_m, idct_m = util.get_dct_matrix(t_pred + t_his)
        dct_m_all = dct_m.float().to(device)
        idct_m_all = idct_m.float().to(device)
        parts = cfg.nf_specs['parts']
        n_parts = len(parts)
        idx_pad = list(range(t_his)) + [t_his - 1] * t_pred
        k = 1
        for traj_np, traj_multimodal_np in generator:
            traj_np = traj_np[..., 1:, :].transpose([0, 2, 3, 1])  # .reshape(traj_np.shape[0], traj_np.shape[1], -1)
            traj = tensor(traj_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()
            bs, nj, _, _ = traj.shape
            inp = traj.reshape([bs, -1, t_his + t_pred]).transpose(1, 2)
            inp = torch.matmul(dct_m_all[:cfg.n_pre], inp[:, idx_pad, :]).transpose(1, 2). \
                reshape([bs, nj, 3, -1]).reshape([bs, nj, -1])

            traj_multimodal_np = traj_multimodal_np[..., 1:, :]  # [bs, modality, seqn, jn, 3]
            traj_multimodal_np = traj_multimodal_np.reshape([bs, n_modality, t_his + t_pred, -1]).transpose(
                [2, 0, 1, 3])
            traj_multimodal = tensor(traj_multimodal_np, device=device, dtype=dtype)  # .permute(0, 2, 1).contiguous()

            inp = inp.unsqueeze(1).repeat([1, (cfg.nk ** n_parts), 1, 1]).reshape(
                [bs * (cfg.nk ** n_parts), nj, -1])
            z = None
            for _ in range(n_parts):
                if z is None:
                    zt = torch.randn([bs, cfg.nk, 1, cfg.nf_specs['nz']], dtype=dtype, device=device)
                    z = zt
                else:
                    z = z.repeat_interleave(cfg.nk, dim=1)
                    zt = torch.randn([bs, z.shape[1], 1, cfg.nf_specs['nz']], dtype=dtype, device=device)
                    z = torch.cat([z, zt], dim=2)
            z = z.reshape([-1, n_parts, cfg.nf_specs['nz']])

            # train generator
            xt = model(inp, z)
            xt = xt.reshape([bs * (cfg.nk ** n_parts), nj, 3, -1]).reshape([bs * (cfg.nk ** n_parts), nj * 3, -1]) \
                .transpose(1, 2)
            traj_est = torch.matmul(idct_m_all[:, :cfg.n_pre], xt).transpose(0, 1)
            traj = traj.reshape([bs, -1, t_his + t_pred]).permute([2, 0, 1])

            # traj
            traj_tmp = traj_est.reshape([-1, traj_est.shape[-1] // 3, 3])
            tmp = torch.zeros_like(traj_tmp[:, :1, :])
            traj_tmp = torch.cat([tmp, traj_tmp], dim=1)
            traj_tmp = util.absolute2relative_torch(traj_tmp, parents=dataset.skeleton.parents()).reshape(
                [-1, traj_est.shape[-1]])
            z, prior_logdetjac = pose_prior(traj_tmp)

            prior_lkh = prior.log_prob(z).sum(dim=-1)
            # prior_logdetjac = log_det_jacobian.sum(dim=2)
            # # normalize traj
            # traj_tmp = (traj_est[t_his::3] - data_mean) / data_std
            # z, log_det_jacobian, _, _, _, _ = pose_prior(traj_tmp)
            #
            # prior_lkh = prior.log_prob(z).sum(dim=2)
            # prior_logdetjac = log_det_jacobian.sum(dim=2)

            loss, losses = loss_function(traj_est, traj, traj_multimodal, prior_lkh, prior_logdetjac)
            train_losses += losses
            total_num_sample += 1
            del loss, z, xt, prior_lkh, prior_logdetjac
    # dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars(name, {'test': loss}, epoch)
    logger.info('====> Epoch: {} Test Time: {:.2f} {} lr: {:.5f}'.format(epoch, time.time() - t_s, losses_str, lr))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',
                        default='h36m')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu_index', type=int, default=1)
    parser.add_argument('--n_pre', type=int, default=8)
    parser.add_argument('--n_his', type=int, default=5)
    parser.add_argument('--trial', type=int, default=1)
    parser.add_argument('--num_coupling_layer', type=int, default=4)
    # parser.add_argument('--nz', type=int, default=10)
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(f'{args.cfg}', test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    cfg.n_his = args.n_his
    if 'n_pre' not in cfg.nf_specs.keys():
        cfg.n_pre = args.n_pre
    else:
        cfg.n_pre = cfg.nf_specs['n_pre']
    cfg.num_coupling_layer = args.num_coupling_layer
    # cfg.nz = args.nz
    """data"""
    if 'actions' in cfg.nf_specs.keys():
        act = cfg.nf_specs['actions']
    else:
        act = 'all'
    dataset_cls = DatasetH36M if cfg.dataset == 'h36m' else DatasetHumanEva
    dataset = dataset_cls('train', t_his, t_pred, actions=act, use_vel=cfg.use_vel,
                          multimodal_path=cfg.nf_specs[
                              'multimodal_path'] if 'multimodal_path' in cfg.nf_specs.keys() else None,
                          data_candi_path=cfg.nf_specs[
                              'data_candi_path'] if 'data_candi_path' in cfg.nf_specs.keys() else None)
    dataset_test = dataset_cls('test', t_his, t_pred, actions=act, use_vel=cfg.use_vel,
                               multimodal_path=cfg.nf_specs[
                                   'multimodal_path'] if 'multimodal_path' in cfg.nf_specs.keys() else None,
                               data_candi_path=cfg.nf_specs[
                                   'data_candi_path'] if 'data_candi_path' in cfg.nf_specs.keys() else None)
    if cfg.normalize_data:
        dataset.normalize_data()

    """model"""
    # model = get_vae_model(cfg, dataset.traj_dim)
    model, pose_prior = get_model(cfg, dataset.traj_dim // 3, args.cfg)
    model.float()
    pose_prior.float()
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    logger.info(">>> total params: {:.2f}M".format(
        sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    cp_path = 'results/h36m_nf/models/vae_0025.p' if cfg.dataset == 'h36m' else 'results/humaneva_nf/models/vae_0025.p'
    print('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    pose_prior.load_state_dict(model_cp['model_dict'])
    pose_prior.to(device)
    # data_mean = tensor(model_cp['meta']['mean'], dtype=dtype, device=device).reshape([-1])
    # data_std = tensor(model_cp['meta']['std'], dtype=dtype, device=device).reshape([-1])

    valid_ang = pickle.load(open('./data/h36m_valid_angle.p', "rb")) if cfg.dataset == 'h36m' else pickle.load(
        open('./data/humaneva_valid_angle.p', "rb"))
    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        overall_iter = 0
        for i in range(args.iter, cfg.num_vae_epoch):
            train(i)
            # val(i)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = cfg.vae_model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))
