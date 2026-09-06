# 仓库规范与数据防误提交

远程仓库：https://github.com/healyops-blip/Stereo-Calibration 。数据集仅保存在本机 dataset/，不通过 Git 分发。

## 数据保护

- `.gitignore` 忽略 dataset/、旧 kaoti/、outputs/、虚拟环境、BMP 与常见压缩包。
- `.githooks/pre-commit` 检查 Git 索引，因此也会拦截 `git add -f` 强制暂存的数据。
- `.githooks/pre-push` 检查索引和全部本地可达历史，防止“先提交数据、再删除”仍把数据留在历史中。
- GitHub Actions 再次检查数据策略，并执行规范检查与合成测试；CI 不使用真实数据集。

每次新克隆仓库后执行：

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit .githooks/pre-push
python3 scripts/check_repository.py --history
```

钩子依赖本地 Python 3，优先使用 .venv/bin/python。没有依赖库也可以执行数据策略检查。
忽略规则和钩子可以被主动绕过，按路径/扩展名的检查不能识别任意改名后的数据。
CI 在文件上传后才运行，不能撤回已经上传的数据；不是服务端上传防火墙。
不得通过关闭钩子或 --no-verify 上传数据。如果数据已进入历史，应停止推送并专门清理历史。

## 代码规范

`.editorconfig` 统一 UTF-8、缩进与换行；`.gitattributes` 统一代码换行。
PR 模板要求说明变更、测试及文档同步。运行：

```bash
python check_quality.py
python scripts/check_repository.py --history
```

CI 工作流为 Repository quality，检查项 Code and data policy。
合成测试不依赖 dataset/；有数据时再本地运行完整精度流程。
GitHub 分支保护是否可强制执行，受仓库权限和账号套餐限制；以 GitHub 实际设置为准。
