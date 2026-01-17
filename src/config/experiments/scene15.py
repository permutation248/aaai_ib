from config.defaults import Experiment, DDC, Fusion, MLP, Loss, Dataset, MSDIB, Optimizer

# Scene15 配置
scene15 = Experiment(
    dataset_config=Dataset(name="scene15"),  # 对应 data/processed/scene15.npz
    model_config=MSDIB(
        backbone_configs=(
            # ---在此处修改输入维度---
            # 请填入 preprocess_scene15.py 运行后打印的 View 0 和 View 1 的特征维度
            # 例如：如果 View 0 是 20维，View 1 是 59维：
            MLP(input_size=(20,)), 
            MLP(input_size=(59,)), 
        ),
        # 你的代码只加载了 X1, X2，所以这里是 2 个视图
        fusion_config=Fusion(method="weighted_mean", n_views=2),
        projector_config=None,
        # Scene15 有 15 个类别
        cm_config=DDC(n_clusters=15),
        loss_config=Loss(
            # 损失函数中的聚类数也需要设为 15
            n_clusters=15,
            funcs="ddc_1|ddc_2|ddc_3|MASG",
            delta=20.0
        ),
        optimizer_config=Optimizer(scheduler_step_size=50, scheduler_gamma=0.1)
    ),
    # 训练参数，可根据需要调整
    n_epochs=100,
    batch_size=100,
)