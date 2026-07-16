#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#
# 持久化启动 safety_gym_env 调试实验。
#
# 设计目标:
#   1. 训练进程脱离当前 SSH 会话: nohup 忽略 SIGHUP, setsid 创建独立 session；
#   2. 不用脆弱的双层 bash -lc 字符串: 参数先按 shell-safe 的 %q 写入独立 run.sh；
#   3. 每个 job 保存 PID、完整命令、Git commit、开始/结束时间和退出码；
#   4. 拒绝覆盖仍在运行的同名 job，避免两个实验写入同一日志/JSON。
#
# 用法:
#   ./launch_background.sh <job_name> -- <command> [args ...]
#
# 示例:
#   ./launch_background.sh dqcac_e1 -- python -u run_experiment.py --algo DQCAC ...

set -euo pipefail

# ===== 1. 解析参数，并保留 `--` 后面的原始 argv 边界 =====
if [[ $# -lt 3 || "$2" != "--" ]]; then
    echo "usage: $0 <job_name> -- <command> [args ...]" >&2
    exit 2
fi

job_name="$1"                                              # 稳定 job 标识，也是默认日志文件名
shift 2                                                     # 删除 job_name 与分隔符 `--`

# job 名只允许安全字符，防止误写到 _runs/jobs 以外的位置。
if [[ ! "$job_name" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "invalid job_name: use only letters, digits, dot, underscore, hyphen" >&2
    exit 2
fi

# ===== 2. 建立实验管理目录；训练产物仍由 run_experiment.py 写入 _runs =====
base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # safety_gym_env 绝对路径
storage_root="$(cd "$base_dir/../../.." && pwd)"          # /vepfs 用户目录，不占 20G 根分区
scratch_dir="${SAFETY_GYM_TMPDIR:-$storage_root/.tmp/safety_gym_env}"
wandb_cache_dir="$scratch_dir/wandb-cache"
mpl_config_dir="$scratch_dir/matplotlib"
job_dir="$base_dir/_runs/jobs/$job_name"                   # 本 job 的控制面文件目录
log_dir="$base_dir/_runs/logs"                             # 与 W&B 导出脚本约定一致
log_file="$log_dir/$job_name.log"                          # stdout/stderr 完整日志
pid_file="$job_dir/pid"                                    # nohup/setsid 外层进程 PID
run_script="$job_dir/run.sh"                               # %q 序列化后的实际运行脚本

mkdir -p "$job_dir" "$log_dir" "$scratch_dir" "$wandb_cache_dir" "$mpl_config_dir"

# 已有 live PID 时拒绝复用名称；空文件或已退出 PID 视为 stale，可安全覆盖控制面文件。
if [[ -s "$pid_file" ]]; then
    old_pid="$(tr -d '[:space:]' < "$pid_file")"
    if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
        echo "job '$job_name' is already running with pid=$old_pid" >&2
        exit 3
    fi
fi

# 同名 stale job 可以复用，但上一轮的完成标记不能保留到新进程启动之后。
# 否则外部轮询器可能在新 worker 尚未结束时读到旧 exit_code，误判本轮已经失败
# 或成功。只清理由本 launcher 生成的三个状态文件，不触碰旧日志与实验产物；
# 新 run.sh 会分别在 worker 真正开始、结束时原子式重建这些文件。
rm -f "$job_dir/worker_started_at" "$job_dir/finished_at" "$job_dir/exit_code"

# ===== 3. 把 argv 安全地写入独立脚本，并让脚本自行记录真实退出状态 =====
{
    printf '#!/usr/bin/env bash\n'
    printf 'set +e\n'                                        # 即使训练失败也必须写 exit_code
    printf 'cd %q\n' "$base_dir"
    printf 'export TMPDIR=%q WANDB_CACHE_DIR=%q MPLCONFIGDIR=%q\n' "$scratch_dir" "$wandb_cache_dir" "$mpl_config_dir"
    printf '/usr/bin/date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ > %q\n' "$job_dir/worker_started_at"
    printf '%q ' "$@"                                       # 每个 argv 用 Bash %q 转义，保留参数边界
    printf '\n'
    printf 'rc=$?\n'
    printf 'printf "%%s\\n" "$rc" > %q\n' "$job_dir/exit_code"
    printf '/usr/bin/date -u +%%Y-%%m-%%dT%%H:%%M:%%SZ > %q\n' "$job_dir/finished_at"
    printf 'exit "$rc"\n'
} > "$run_script"
chmod 700 "$run_script"

# 另外保存人类可读的一行命令，便于文档记录和完全复现。
printf '%q ' "$@" > "$job_dir/command.txt"
printf '\n' >> "$job_dir/command.txt"

# 记录启动时版本；dirty 状态单独记录，避免误以为实验必然来自干净 commit。
git -C "$base_dir" rev-parse HEAD > "$job_dir/git_commit"
git -C "$base_dir" status --short -- . > "$job_dir/git_status.txt"
/usr/bin/date -u +%Y-%m-%dT%H:%M:%SZ > "$job_dir/started_at"

# ===== 4. 脱离 SSH 启动：stdout/stderr 都落盘，stdin 关闭 =====
# setsid 创建新的 session/process group；nohup 让进程忽略终端挂断信号。
nohup setsid /bin/bash "$run_script" > "$log_file" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pid_file"

# 启动后做轻量存活检查；训练初始化较慢时日志为空是正常现象。
sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
    echo "job '$job_name' exited during startup; inspect $log_file" >&2
    exit 4
fi

echo "started job=$job_name pid=$pid"
echo "log=$log_file"
echo "job_dir=$job_dir"
