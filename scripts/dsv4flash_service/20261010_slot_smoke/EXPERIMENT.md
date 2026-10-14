# 实验：首次 slot 服务连通性验证 · 2026-10-10

环境重建后首次用 `../service/start_service_mode_dp4.sh slot` 起 DSV4-Flash DP4 服务
（端口 8444，model 名 test_auto），验证 MoonEP slot 初始化链路（shmem ipport 引导、
home/slot 镜像、real-T、slot 量化）与推理请求可通。日志：`start_service_dp4_slot.log`
（`[moonep-init]`、`[moonep-release]` 等关键行都在其中）。
