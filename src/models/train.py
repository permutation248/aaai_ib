import torch.utils.data
import wandb
import torch as th

import numpy as np
import random
import config
import helpers
from data.load import load_dataset
from models import callback
from models.build_model import build_model
from models import evaluate
from config.defaults import Loss
import time
from models.evaluate import calc_metrics


def metrics_loss_acc_nmi(net, eval_data, batch_size):
    losses = []
    predictions = []
    labels = []
    net.eval()
    for i, (*batch, label, _) in enumerate(eval_data):
        if label.size(0) == batch_size:

            pred, _, _ = net(batch)

            predictions.append(helpers.npy(pred).argmax(axis=1))
            labels.append(helpers.npy(label))


            batch_losses = net.calc_losses(ignore_in_total='')

            losses.append(helpers.npy(batch_losses))

    net.train()
    labels = np.concatenate(labels, axis=0)
    predictions = np.concatenate(predictions, axis=0)
    metrics = calc_metrics(labels, predictions)
    acc = metrics['acc']
    nmi = metrics['nmi']
    losses = helpers.dict_means(losses)

    return losses['tot'], acc, nmi


def list_txt(path, list=None):

    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist


def train(cfg, net, loader, run, eval_data, callbacks=tuple()):
    """
    Train the model for one run.

    :param cfg: Experiment config
    :type cfg: config.defaults.Experiment
    :param net: Model
    :type net:
    :param loader: DataLoder for training data
    :type loader:  th.utils.data.DataLoader
    :param eval_data: DataLoder for evaluation data
    :type eval_data:  th.utils.data.DataLoader
    :param callbacks: Training callbacks.
    :type callbacks: List
    :return: None
    :rtype: None
    """
    n_batches = len(loader)
    Loss_list = []
    Accuracy_list = []
    NMI_list = []
    time_list = []
    begin = time.time()
    max_acc = 0
    for e in range(1, cfg.n_epochs + 1):
        iter_losses = []
        for i, data in enumerate(loader):
            *batch, _, index = data

            try:
                #batch为传入的数据，epoch为训练多少轮，it为一轮的第几个batch，n_batches为一轮有多少个batch
                batch_losses = net.train_step(batch, epoch=(e-1), it=i, n_batches=n_batches)
            except Exception as e:
                print(f"Training stopped due to exception: {e}")
                return

            iter_losses.append(helpers.npy(batch_losses))
        logs = evaluate.get_logs(cfg, net, eval_data=eval_data, iter_losses=iter_losses, epoch=e, include_params=True)
        if (e is None) or ((e % cfg.eval_interval) == 0):
            loss, acc, nmi = metrics_loss_acc_nmi(net, eval_data, batch_size=100)
            Loss_list.append(loss)
            Accuracy_list.append(acc)
            NMI_list.append(nmi)
            if acc > max_acc:
                max_acc = acc

        try:
            for cb in callbacks:
                cb.epoch_end(e, logs=logs, net=net)
        except callback.StopTraining as err:
            print(err)
            break

        end = time.time()
        time_list.append(end - begin)


    list_txt(path='../../acc_nmi_loss/iapr_loss.txt', list=Loss_list)
    list_txt(path='../../time_epoch/iapr_time.txt', list=time_list)
    list_txt(path='../../acc_nmi_loss/iapr_acc.txt', list=Accuracy_list)
    list_txt(path='../../acc_nmi_loss/iapr_nmi.txt', list=NMI_list)


def main():
    """
    Run an experiment.
    """
    experiment_name, cfg = config.get_experiment_config()
    """     # 实验名称
    experiment_name='iapr'

    # 配置文件 (cfg)
    cfg={
        # 数据集配置
        'dataset_config': Dataset(
            name='iapr',               # 数据集名称
            n_samples=None,            # 样本数量 (None表示使用全部)
            select_views=None,         # 选择的视图 (None表示使用全部)
            select_labels=None,        # 选择的标签 (None表示使用全部)
            label_counts=None,         # 标签计数 (None表示不指定)
            noise_sd=None,             # 噪声标准差 (None表示无噪声)
            noise_views=None           # 施加噪声的视图 (None表示无噪声)
        ),

        # 模型配置 (MSDIB: Multi-View Self-Supervised Deep Information Bottleneck?)
        'model_config': MSDIB(
            # 骨干网络配置 (多视图，这里是2个视图)
            'backbone_configs': (
                MLP(                       # 视图1的MLP
                    input_size=(100,),     # 输入维度
                    layers=(512, 512, 256),# 隐藏层大小
                    activation='relu',     # 激活函数
                    use_bias=True,         # 使用偏置
                    use_bn=True            # 使用批量归一化 (Batch Norm)
                ),
                MLP(                       # 视图2的MLP
                    input_size=(100,),
                    layers=(512, 512, 256),
                    activation='relu',
                    use_bias=True,
                    use_bn=True
                )
            ),
            'projector_config': None,      # 投影器配置 (None表示不使用)
            
            # 融合配置
            'fusion_config': Fusion(
                method='weighted_mean',    # 融合方法: 加权平均
                n_views=2                  # 视图数量
            ),
            
            # 聚类模块配置 (CM: Clustering Module, DDC: Deep Divergence Clustering?)
            'cm_config': DDC(
                n_clusters=6,              # 聚类簇数
                n_hidden=100,              # 隐藏层大小
                use_bn=True                # 使用批量归一化
            ),
            
            # 损失函数配置
            'loss_config': Loss(
                n_clusters=6,
                funcs='ddc_1|ddc_2|ddc_3|MASG', # 损失函数项
                weights=None,              # 损失项权重 (None表示默认或平均)
                negative_samples_ratio=0.25,# 对比学习负样本比例
                contrastive_similarity='cos',# 对比相似度度量: 余弦相似度
                rel_sigma=0.15,            # 相关性度量参数 $\sigma$
                tau=0.1,                   # 温度参数 $\tau$ (用于对比学习?)
                delta=20.0,                # 另一个损失项参数 $\delta$
                adaptive_contrastive_weight=True, # 使用自适应对比权重
                c1=0.8, c2=0.6             # 自适应权重相关的系数
            ),
            
            # 优化器配置
            'optimizer_config': Optimizer(
                learning_rate=0.0001,      # 学习率
                clip_norm=5.0,             # 梯度裁剪范数
                scheduler_step_size=50,    # 学习率调度器步长
                scheduler_gamma=0.1        # 学习率调度器乘数
            )
        ),

        # 训练/评估参数
        'eval_interval': 4,                # 评估间隔 (每隔4个epoch)
        'n_eval_samples': None,            # 评估样本数 (None表示使用全部)
        'n_runs': 20,                      # 重复运行次数
        'n_epochs': 100,                   # 总训练轮数
        'batch_size': 100,                 # 批次大小
        'checkpoint_interval': 20,         # 检查点保存间隔 (每隔20个epoch)
        'patience': 50000,                 # 早停耐心值 (值非常大，相当于不使用早停?)
        'best_loss_term': 'ddc_1',         # 用于确定最佳模型的损失项
        'seed': 0                          # 随机种子
    } """
    dataset = load_dataset(**cfg.dataset_config.dict(), n_views=cfg.model_config.fusion_config.n_views)

    loader = th.utils.data.DataLoader(dataset, batch_size=int(cfg.batch_size), shuffle=True, num_workers=0,
                                      drop_last=True, pin_memory=False)

    eval_data = evaluate.get_eval_data(dataset, cfg.n_eval_samples, cfg.batch_size)
    experiment_identifier = wandb.util.generate_id()
    print(experiment_identifier)

    run_logs = []
    for run in range(cfg.n_runs):
        net = build_model(cfg.model_config)
        callbacks = (
            callback.Printer(print_confusion_matrix=(cfg.model_config.cm_config.n_clusters <= 100)),
            callback.ModelSaver(cfg=cfg, experiment_name=experiment_name, identifier=experiment_identifier,
                                run=run, epoch_interval=1, best_loss_term=cfg.best_loss_term,
                                checkpoint_interval=cfg.checkpoint_interval),
            callback.EarlyStopping(patience=cfg.patience, best_loss_term=cfg.best_loss_term, epoch_interval=1)
        )

        train(cfg, net, loader, run, eval_data=eval_data, callbacks=callbacks)

        run_logs.append(evaluate.eval_run(cfg=cfg, cfg_name=experiment_name,
                                          experiment_identifier=experiment_identifier, run=run, net=net,
                                          eval_data=eval_data, callbacks=callbacks, load_best=True))





if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False
    print(seed)
    main()

