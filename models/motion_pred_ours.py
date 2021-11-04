from models import GCN, LinNF


def get_model(cfg, traj_dim, model_type='h36m_naf'):

    if model_type == 'h36m' or model_type == 'humaneva':
        data_dim = cfg.n_pre * 3
        return GCN.GCNParts(input_feature=data_dim + cfg.nf_specs['nz'], hidden_feature=cfg.nf_specs['hidden_dim'],
                               is_bn=True, parts=cfg.nf_specs['parts'], num_stage=cfg.nf_specs['num_stage'],
                               node_n=traj_dim), \
               LinNF.LinNF(data_dim=traj_dim * 3, num_layer=3)

    elif model_type == 'h36m_nf' or model_type == 'humaneva_nf':
        return LinNF.LinNF(data_dim=traj_dim, num_layer=cfg.nf_specs['num_flow_layer'])