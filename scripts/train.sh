#!/bin/bash

# Pre-start
# checkmark font for fancy log
CHECK_MARK="\033[0;32m\xE2\x9C\x94\033[0m"
# usage text
USAGE="$(basename "$0") CONFIG_NAME [-h] [-d] [-p PRECISION] [-c] [-m GPU_MEM] [-o]

where:
    CONFIG_NAME   Configuration name (one of the configuration files in .config)
    -h            Show this help text
    -d            Run in debug mode (no GPU and wandb offline)
    -p            Training precision, default 16.
    -c            Use CPU instead of GPU.
    -g            Which GPU to use, default [0].
    -m            Minimum GPU memory required in MB (default: 8000). If less that this,
                  training will wait until there is enough space.
    -o            Run the experiment offline

Example:
  ./script/train.sh root
  ./script/train.sh root -m 10000
"

# check for named params
while [ $OPTIND -le "$#" ]; do
  if getopts ":hdp:cgm:o" opt; then
    case $opt in
    h)
      printf "%s$USAGE" && exit 0
      ;;
    d)
      DEV_RUN="True"
      ;;
    p)
      PRECISION="$OPTARG"
      ;;
    c)
      USE_CPU="True"
      ;;
    g)
      GPUS="[$OPTARG]"
      ;;
    m)
      GPU_MEM="$OPTARG"
      ;;
    o)
      WANDB="offline"
      ;;
    \?)
      echo "Invalid option -$OPTARG" >&2 && echo "$USAGE" && exit 0
      ;;
    esac
  else
    CONFIG_NAME=${*:$OPTIND:1}
    ((OPTIND++))
  fi
done

# PRELIMINARIES
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/bin/activate potto

## if CONFIG_NAME is not specified, abort
if [ -z "$CONFIG_NAME" ]; then
  printf "A configuration name must be specified.\n\n"
  printf "%s$USAGE"
  exit 0
fi

# if -d is not specified, DEV_RUN is False
if [ -z "$DEV_RUN" ]; then
  # default value
  DEV_RUN=False
fi

# if -p is not specified, PRECISION is 16
if [ -z "$PRECISION" ]; then
  # default value
  PRECISION=16
fi

# if -c is not specified, USE_CPU is False
if [ -z "$USE_CPU" ]; then
  # default value
  USE_CPU=False
fi

# if -g is not specified, GPUS is 1
if [ -z "$GPUS" ]; then
  # default value
  GPUS="[0]"
fi

# if -m is not specified, GPU_MEM is not limited
if [ -z "$GPU_MEM" ]; then
  # default value
  GPU_MEM=0
fi

# if -o is not specified, WANDB is "online"
if [ -z "$WANDB" ]; then
  # default value
  WANDB="online"
fi

# CHECK FOR BOOLEAN PARAMS
# if -d then GPU is not required and no output dir
if [ "$DEV_RUN" = "True" ]; then
  WANDB="offline"
fi

# if -c GPUS is 0 (no GPU) and PRECISION is 32
if [ "$USE_CPU" = "True" ]; then
  # default value
  GPUS=0
  PRECISION=32
fi

if type nvidia-smi >/dev/null 2>&1; then
  FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo "[0-9]+")
else
  FREE_MEM=0
  GPU_MEM=0
  GPUS=0
  PRECISION=32
  if [ $USE_CPU = "False" ]; then
    echo -e "GPU not found, fallback to CPU.\n"
  fi
  USE_CPU="True"
fi
GPU_RAM_MESSAGE=""

# echo configuration

cat <<EOF
Configuration:
------------------------------------------------
Configuration name:   $CONFIG_NAME
Run in debug mode:    $DEV_RUN
Requested VRAM:       $GPU_MEM MB
Available VRAM:       $FREE_MEM MB
Precision:            $PRECISION bit
Use CPU:              $USE_CPU
W&B Mode:             $WANDB

EOF

# WAITING FOR VRAM STUFF
chars="/-\|"
if [ "$FREE_MEM" -lt $GPU_MEM ]; then
  # echo -n "$GPU_RAM_MESSAGE"
  GPU_RAM_MESSAGE="\\rWaiting for at least $GPU_MEM MB of VRAM "
  echo -ne "$GPU_RAM_MESSAGE"
fi
while [ "$FREE_MEM" -lt $GPU_MEM ]; do
  FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0 | grep -Eo "[0-9]+")
  for ((i = 0; i < ${#chars}; i++)); do
    sleep 1
    echo -ne "${chars:$i:1} $GPU_RAM_MESSAGE"
  done
done

# if DEV_RUN then GPU is not required
if [ "$DEV_RUN" = "True" ]; then
  echo -n "Debug run started, ignoring GPU memory. "
  GPU_RAM_MESSAGE=""
fi

echo -e "$GPU_RAM_MESSAGE${CHECK_MARK} Starting.\n"

# if you use the `GenerativeDataset` class
# you may want to set `TOKENIZERS_PARALLELISM` to `false`
#export TOKENIZERS_PARALLELISM=false

if [ "$DEV_RUN" = "True" ]; then
  export HYDRA_FULL_ERROR=1
  python src/train.py --config-name="$CONFIG_NAME" \
    "train.pl_trainer.fast_dev_run=$DEV_RUN" \
    "train.pl_trainer.gpus=$GPUS" \
    "train.pl_trainer.precision=$PRECISION" \
    "hydra.run.dir=." \
    "hydra.output_subdir=null"
else
  python src/train.py --config-name="$CONFIG_NAME" \
    "train.pl_trainer.fast_dev_run=$DEV_RUN" \
    "train.pl_trainer.gpus=$GPUS" \
    "train.pl_trainer.precision=$PRECISION" \
    "logging.wandb_arg.mode=$WANDB"
fi
