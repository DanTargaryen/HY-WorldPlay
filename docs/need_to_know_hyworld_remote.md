# Need To Know — HY-WorldPlay Remote Execution

This file contains the minimum operational rules a remote agent must follow
before running any HY-WorldPlay task.

## 1. Where Code, Models, and Data Live

### Code and model / weight root

```bash
PROJECT_ROOT=/mnt/pfs/users/huangzehuan/projects/linming/HY-WorldPlay
MODEL_ROOT=/mnt/pfs/users/huangzehuan/projects/linming
```

### Data and output root

```bash
DATA_ROOT=/mnt/pfs/data/huangzehuan/datasets
TRACE_ROOT=/mnt/pfs/data/huangzehuan/datasets/hyworld_ar_distill_context_traces
OUTPUT_ROOT=/mnt/pfs/data/huangzehuan/datasets/hyworld_ar_distill_outputs
LOG_ROOT=/mnt/pfs/data/huangzehuan/datasets/hyworld_ar_distill_logs
REPORT_ROOT=/mnt/pfs/data/huangzehuan/datasets/hyworld_ar_distill_reports
```

## 2. How To Use GPU

Do real GPU work inside these Kubernetes pods:

```bash
kubectl exec -it hzh-dev-juicef-2n-master-0 -- bash
kubectl exec -it hzh-dev-juicef-2n-worker-0 -- bash
```

Do not assume the current shell already has GPU access.

Inside the pod, verify GPU visibility with:

```bash
nvidia-smi
```

## 3. Always Use tmux

Before any long-running task, start or attach a tmux session:

```bash
tmux new -s hyworld || tmux attach -t hyworld
```

This applies to:

- model downloads
- long trace generation
- AR-distill inference
- any smoke or full-length run

## 4. Network Proxy

Before any network-dependent action, export:

```bash
export http_proxy=http://192.168.48.17:18000
export https_proxy=http://192.168.48.17:18000
```

Apply this before:

- `pip install`
- `git pull`
- HuggingFace downloads
- any remote API or model fetch

## 5. Conda Activation

Before any python / pip / torch / torchrun command, run:

```bash
source /mnt/pfs/users/huangzehuan/dev/miniconda3/bin/activate
```

Then inspect available environments if needed:

```bash
conda env list
```

If there is a project-specific environment, activate it:

```bash
conda activate <env_name>
```

## 6. Absolute Path Rule

Any path beginning with `/` is an absolute path and must be used exactly as
written.

Do not prepend `PROJECT_ROOT` or current working directory to absolute paths.

Example:

- correct: `/mnt/pfs/users/huangzehuan/dev/miniconda3/bin/activate`
- wrong: `/mnt/pfs/users/huangzehuan/projects/linming/users/huangzehuan/dev/miniconda3/bin/activate`

## 7. Minimum Startup Sequence

A safe startup sequence is:

```bash
kubectl exec -it hzh-dev-juicef-2n-master-0 -- bash
# or worker pod

tmux new -s hyworld || tmux attach -t hyworld

export http_proxy=http://192.168.48.17:18000
export https_proxy=http://192.168.48.17:18000

source /mnt/pfs/users/huangzehuan/dev/miniconda3/bin/activate
conda env list

cd /mnt/pfs/users/huangzehuan/projects/linming/HY-WorldPlay
```

## 8. Task Entry

For the current AR-distill memory baseline task, read:

```bash
/mnt/pfs/users/huangzehuan/projects/linming/HY-WorldPlay/TODO_hyworld_ar_distill_memory.md
```

Then execute the task exactly as specified there.
