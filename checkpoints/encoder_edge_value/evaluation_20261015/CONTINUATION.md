# 本轮结果后的明确续训决定

2026-10-15，用户在获知早期最佳权重 optimized solid 为2,118/2,417、最后14,520步权重为2,095/2,417（旧conditioning为2,111）后，明确选择：**“仍从最后权重按原配置继续训练”**。这项授权取代先前“有改善才续训”的条件分歧；不将最后权重描述为已经改善。

恢复候选 `candidate/20261015T073112.923985Z` 的最后完整checkpoint，epoch39、global_step14,520，SHA256 `e9120c78254d9b94aaaef88333c0f0656fd6a01346225d092fb8341297bff2bf`。新增预算14,520 optimizer steps（40个完整epoch），累计上限29,040 steps／80 epochs；原GAT对照不自动排队。

严格恢复同一run的权重、Adam状态、训练loop与W&B身份。保持原配置：8 GPU×每卡batch16、lr2e-5、bf16、seed42、Encoder/posterior/Decoder全量解冻；边内容MLP、intersection、refinement、curve-conditioning、KL及其他loss权重、数据与augmentation均不改变。point-feedback仍关闭。只调整总预算与resume相关路径，不重新初始化模型或优化器。

启动器先把本次评估的1,089步最佳权重及14,520步最后权重保存为不可变归档，并记录原路径→归档路径与SHA；不修改已冻结的评估协议。继续训练会更新原run的last.ckpt，但可通过归档重现本轮结果。使用已有 `run_pair.py` 的显式单组resume入口和现有status/stop入口，不增加trainer。

已验证归档在同一run的 `weights/evaluated_snapshots/`，文件名为对应SHA256加`.ckpt`。评估入口脚本后续重现时明确指向这两份相同字节的归档；历史调用日志、推理协议和训练审计仍保留当时的原始路径，不回写历史记录。原GAT对照没有续训，保留原权重路径。

```bash
/mnt/d/anaconda3/envs/torch/bin/python /mnt/d/python_curve_eval/checkpoints/encoder_edge_value/run_pair.py resume --arm candidate --max-steps 29040 --max-epochs 80
```

```bash
/mnt/d/anaconda3/envs/torch/bin/python /mnt/d/python_curve_eval/checkpoints/encoder_edge_value/run_pair.py status
/mnt/d/anaconda3/envs/torch/bin/python /mnt/d/python_curve_eval/checkpoints/encoder_edge_value/run_pair.py stop
```

实际启动、恢复与进度状态以相邻实验的 `metadata/active_pair.json`、新resume日志和本目录最终REPORT为准。停止命令核验进程身份并发送中断；最近保存的checkpoint只保证包含最近完成的保存点。

已于2026-10-15 13:35 UTC启动，launch ID为`20261015T133540.963165Z`，继续同一W&B run `376247fb7478`。13:44 UTC验证首个完整续训epoch：step14,883／epoch40，`val/loss=0.0420773`，`last.ckpt`保存成功并进入epoch41。两份归档SHA、8个rank、优化器参数组和756个参数状态均通过核验；实际resolved config仅发生上文批准的四项变化。完整记录见 [resume_validation.json](../validation/resume_validation.json)。训练仍在运行，不等待全部新增预算结束才交付本轮评估。
