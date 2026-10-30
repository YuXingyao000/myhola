# IC-Light 背景条件重光照：模型准备

## Task
为下一轮验证准备公开SD1.5背景条件重光照模型；当前只下载/校验，不声称已运行或有效。
动机是两种程序金属表面只提亮，没有改善层次；现有背景增强已可独立复用。

## Assumptions
只用本地算力与免费公开权重，不调用收费API，不下载BRIA（已有真CAD mask），
不覆盖generate环境依赖。生成式重光照可能改变CAD，之后必须另设保真比较。
代码Apache-2.0；模型卡声明CreativeML OpenRAIL-M。HF ic-light元数据license缺失，
保存实际README原文，不把权重错误标成Apache或无条件自由使用。

## Inputs
官方代码bcf3f29ca85be8a4686215f477b546f5030be8b7；
lllyasviel/ic-light revision 9cad1878695f546a7fb9eaca14e2a89131ba5ffe；
stablediffusionapi/realistic-vision-v51 revision 19e3643d7d963c156d01537188ec08f0b79a514a。
只取fbc差分权重及base tokenizer/text_encoder/VAE/UNet/scheduler配置和safetensors。

## Outputs
`/mnt/d/model/ICLight_SD15/`：base/、relight/、官方原始demo/README/许可、API元数据与SHA清单。
不执行下载的demo，不安装其旧requirements，不开启Gradio服务。

## Protocol
按固定revision读HF带blobs元数据，逐文件写.partial，核对大小及LFS SHA256后重命名；
完成后保存source.json。核心推理代码与六件运行协议另行冻结，再执行实际模型加载和推理。
当前环境仅完成核心类CPU导入，不等于权重加载/采样兼容已验证。

## Edge Cases
网络或摘要错误直接失败、保留partial；不静默换源、换revision或忽略下载失败。
复用已有路径时先核查状态，不覆盖历史模型来源。后台下载必须核对真实进程与终态。
