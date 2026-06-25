# setup_b200.sh — TRELLIS env for NVIDIA B200 (Blackwell, sm_100).
#
# Differs from setup.sh:
#   - PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 (cu124 builds cannot emit
#     sm_100 kernels for B200; modern xformers also requires torch >= 2.10, so
#     the older 2.7.0 pin is no longer viable).
#   - CUDA 12.8 toolkit + gcc 12 installed inside the conda env, so we don't
#     depend on the host's /usr/local/cuda version.
#   - All CUDA extensions built with TORCH_CUDA_ARCH_LIST="10.0" so they emit
#     B200-compatible kernels.
#   - cumm v0.7.11 built from source with CUMM_DISABLE_JIT=1. Why:
#       * spconv 2.3.x pins `cumm-cu128>=0.7.11,<0.8.0`, but PyPI only
#         publishes cumm-cu128 0.8.2 — so the prebuilt wheel is unusable.
#       * Without CUMM_DISABLE_JIT=1, the source build is Python-only (~370KB
#         wheel, no `cumm/core_cc*.so`) and spconv's setup.py crashes with
#         `ModuleNotFoundError: No module named 'cumm.core_cc'`.
#   - --no-deps used aggressively on every post-torch extension install so
#     xformers/flash-attn/cumm/spconv/kaolin can't silently yank torch out
#     from under everything else (which then breaks torchvision's compiled
#     ops with `RuntimeError: operator torchvision::nms does not exist`).
#
# Conda env name: morphos (python=3.10).
#
# Recommended invocation (note --cumm before --spconv):
#   bash setup_b200.sh --new-env --basic --train \
#                      --xformers --flash-attn \
#                      --nvdiffrast --cumm --spconv --kaolin \
#                      --diffoctreerast --mipgaussian
#
# Re-run later with just the option you want to add — `--new-env` will
# recreate the environment from scratch (destructive); omit it to add to an
# existing env.

# Read Arguments
TEMP=`getopt -o h --long help,new-env,basic,train,xformers,flash-attn,diffoctreerast,vox2seq,cumm,spconv,mipgaussian,kaolin,nvdiffrast -n 'setup_b200.sh' -- "$@"`

eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
TRAIN=false
XFORMERS=false
FLASHATTN=false
DIFFOCTREERAST=false
VOX2SEQ=false
CUMM=false
SPCONV=false
ERROR=false
MIPGAUSSIAN=false
KAOLIN=false
NVDIFFRAST=false

if [ "$#" -eq 1 ] ; then
    HELP=true
fi

while true ; do
    case "$1" in
        -h|--help) HELP=true ; shift ;;
        --new-env) NEW_ENV=true ; shift ;;
        --basic) BASIC=true ; shift ;;
        --train) TRAIN=true ; shift ;;
        --xformers) XFORMERS=true ; shift ;;
        --flash-attn) FLASHATTN=true ; shift ;;
        --diffoctreerast) DIFFOCTREERAST=true ; shift ;;
        --vox2seq) VOX2SEQ=true ; shift ;;
        --cumm) CUMM=true ; shift ;;
        --spconv) SPCONV=true ; shift ;;
        --mipgaussian) MIPGAUSSIAN=true ; shift ;;
        --kaolin) KAOLIN=true ; shift ;;
        --nvdiffrast) NVDIFFRAST=true ; shift ;;
        --) shift ; break ;;
        *) ERROR=true ; break ;;
    esac
done

if [ "$ERROR" = true ] ; then
    echo "Error: Invalid argument"
    HELP=true
fi

if [ "$HELP" = true ] ; then
    echo "Usage: setup_b200.sh [OPTIONS]"
    echo "Options:"
    echo "  -h, --help              Display this help message"
    echo "  --new-env               Create a fresh conda env 'morphos' (python=3.10)"
    echo "                          and install torch 2.11.0+cu128 + CUDA 12.8 + gcc 12"
    echo "  --basic                 Install basic dependencies"
    echo "  --train                 Install training dependencies"
    echo "  --xformers              Install xformers (latest, cu128, --no-deps)"
    echo "  --flash-attn            Build flash-attn from source for sm_100"
    echo "  --diffoctreerast        Build diffoctreerast from source"
    echo "  --vox2seq               Build vox2seq from source"
    echo "  --cumm                  Build cumm v0.7.11 from source (REQUIRED before --spconv)"
    echo "  --spconv                Build spconv 2.3.x from source (depends on --cumm)"
    echo "  --mipgaussian           Build mip-splatting (diff-gaussian-rasterization)"
    echo "  --kaolin                Build kaolin from source (no cu128 wheels yet)"
    echo "  --nvdiffrast            Build nvdiffrast from source"
    return 2>/dev/null || exit 0
fi

ENV_NAME="morphos"
TORCH_VER="2.11.0"
TORCHVISION_VER="0.26.0"
CUDA_TAG="cu128"                 # PyTorch wheel index suffix
CUDA_TOOLKIT_VER="12.8"          # cuda-toolkit conda package version
GCC_VER="12"                     # host compiler for nvcc
TARGET_ARCH="${TARGET_ARCH:-10.0}"   # B200 = sm_100; override to "8.0;9.0;10.0" for multi-arch
CUMM_TAG="v0.7.11"               # spconv 2.3.x pins cumm-cu128 <0.8.0; 0.7.11 is the highest 0.7.x
CUMM_ARCH="${CUMM_ARCH:-8.0;8.9;9.0+PTX}"

# ---------------------------------------------------------------------------
# Activate conda
# ---------------------------------------------------------------------------
if ! command -v conda >/dev/null 2>&1 ; then
    echo "[ERROR] conda not on PATH. Source ~/.bashrc / activate miniforge first."
    return 2>/dev/null || exit 1
fi
eval "$(conda shell.bash hook)"

# ---------------------------------------------------------------------------
# --new-env: create env, install CUDA toolkit + gcc, install torch
# ---------------------------------------------------------------------------
if [ "$NEW_ENV" = true ] ; then
    echo "==> Creating fresh conda env: $ENV_NAME (python=3.10)"
    conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" && \
        conda env remove -y -n "$ENV_NAME"
    conda create -y -n "$ENV_NAME" python=3.10
    conda activate "$ENV_NAME"

    echo "==> Installing CUDA $CUDA_TOOLKIT_VER toolkit (env-local)"
    conda install -y -c "nvidia/label/cuda-${CUDA_TOOLKIT_VER}.0" cuda-toolkit="$CUDA_TOOLKIT_VER" \
        || conda install -y -c nvidia "cuda-toolkit=${CUDA_TOOLKIT_VER}"

    echo "==> Installing gcc $GCC_VER (host compiler for nvcc)"
    conda install -y -c conda-forge "gxx_linux-64=$GCC_VER" "gcc_linux-64=$GCC_VER" \
        sysroot_linux-64=2.28 ninja cmake

    echo "==> Wiring up activate.d hook so CUDA_HOME/PATH/CC stay set"
    ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
    DEACT_DIR="$CONDA_PREFIX/etc/conda/deactivate.d"
    mkdir -p "$ACT_DIR" "$DEACT_DIR"
    cat > "$ACT_DIR/morphos_b200_env.sh" <<'EOF'
export _OLD_CUDA_HOME="$CUDA_HOME"
export _OLD_CC="$CC"
export _OLD_CXX="$CXX"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
# ccimport (cumm/spconv build) only overrides the CUDA compiler when CUDACXX is
# set; otherwise it resolves to /usr/local/cuda/bin/nvcc (the host CUDA, which
# may be 13.x and reject the deprecated sm_5x/6x archs). Pin it to env nvcc.
export _OLD_CUDACXX="$CUDACXX"
export CUDACXX="$CONDA_PREFIX/bin/nvcc"
EOF
    cat > "$DEACT_DIR/morphos_b200_env.sh" <<'EOF'
export CUDA_HOME="$_OLD_CUDA_HOME"
export CC="$_OLD_CC"
export CXX="$_OLD_CXX"
export CUDACXX="$_OLD_CUDACXX"
unset _OLD_CUDA_HOME _OLD_CC _OLD_CXX _OLD_CUDACXX
EOF
    # Apply right now in this shell too
    source "$ACT_DIR/morphos_b200_env.sh"

    echo "==> Installing PyTorch $TORCH_VER ($CUDA_TAG)"
    pip install --upgrade pip
    pip install "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" \
        --index-url "https://download.pytorch.org/whl/$CUDA_TAG"

    echo "==> Sanity check"
    python - <<'PY'
import torch
print(f'torch         : {torch.__version__}')
print(f'cuda runtime  : {torch.version.cuda}')
print(f'cuda available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'device        : {torch.cuda.get_device_name(0)}')
    print(f'capability    : {torch.cuda.get_device_capability(0)}')
PY
    import_cap=$(python -c "import torch;print(torch.cuda.get_device_capability(0))" 2>/dev/null || true)
    echo "[INFO] device capability: $import_cap (expected (10, 0) on B200)"
else
    # Re-entering an existing env — make sure it's activated and the toolchain
    # variables are visible.
    conda activate "$ENV_NAME"
fi

# Re-source the activate hook in case we're invoked without --new-env
ACT_HOOK="$CONDA_PREFIX/etc/conda/activate.d/morphos_b200_env.sh"
if [ -f "$ACT_HOOK" ] ; then
    source "$ACT_HOOK"
fi

# ---------------------------------------------------------------------------
# System info (for the per-extension blocks)
# ---------------------------------------------------------------------------
WORKDIR=$(pwd)
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
echo "[SYSTEM] PyTorch $PYTORCH_VERSION, CUDA $CUDA_VERSION, target arch $TARGET_ARCH"

# ---------------------------------------------------------------------------
# Basic / train deps (pure-python, no CUDA build)
# ---------------------------------------------------------------------------
if [ "$BASIC" = true ] ; then
    echo "==> Installing basic dependencies"
    pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless \
        scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph \
        transformers pccm
    pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8
fi

if [ "$TRAIN" = true ] ; then
    echo "==> Installing training dependencies"
    pip install tensorboard pandas lpips
    # pillow-simd needs libjpeg-dev system-wide; skip the apt step on systems
    # where the user doesn't have sudo (just keep stock pillow if it fails).
    pip uninstall -y pillow || true
    pip install pillow-simd || pip install pillow
fi

# ---------------------------------------------------------------------------
# CUDA extensions — every one of these gets TORCH_CUDA_ARCH_LIST=$TARGET_ARCH
# so the resulting .so files include B200 sm_100 cubin.
# ---------------------------------------------------------------------------
PIP_BUILD_ENV="TORCH_CUDA_ARCH_LIST=$TARGET_ARCH MAX_JOBS=${MAX_JOBS:-4}"

if [ "$XFORMERS" = true ] ; then
    echo "==> Installing xformers (latest, $CUDA_TAG, --no-deps)"
    # --no-deps is critical: without it, `pip install -U xformers` will
    # silently bump torch to whatever xformers latest requires (e.g. 2.11),
    # which then mismatches the torchvision wheel ABI and breaks
    # `torchvision::nms` at import time.
    pip install -U --no-deps xformers --index-url "https://download.pytorch.org/whl/$CUDA_TAG" \
        || env $PIP_BUILD_ENV pip install -v --no-deps xformers
fi

if [ "$FLASHATTN" = true ] ; then
    echo "==> Building flash-attn from source (slow; sets MAX_JOBS=${MAX_JOBS:-4})"
    # flash-attn 2.7.x added Hopper/Blackwell. `--no-build-isolation` so the
    # build sees the torch already installed; `--no-deps` so it doesn't
    # silently swap that torch out for the version flash-attn metadata wants.
    env $PIP_BUILD_ENV pip install -v --no-deps flash-attn --no-build-isolation
fi

if [ "$NVDIFFRAST" = true ] ; then
    echo "==> Building nvdiffrast from source"
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/nvdiffrast
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
    env $PIP_BUILD_ENV pip install -v /tmp/extensions/nvdiffrast --no-build-isolation
fi

if [ "$DIFFOCTREERAST" = true ] ; then
    echo "==> Building diffoctreerast from source"
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/diffoctreerast
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
    env $PIP_BUILD_ENV pip install -v /tmp/extensions/diffoctreerast
fi

if [ "$MIPGAUSSIAN" = true ] ; then
    echo "==> Building diff-gaussian-rasterization (mip-splatting) from source"
    pip uninstall -y diff_gaussian_rasterization || true
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/mip-splatting
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
    # --no-build-isolation / --no-deps so the C++ extension links against
    # *our* pinned torch instead of whatever pip's isolated build env pulls.
    env $PIP_BUILD_ENV pip install -v --no-build-isolation --no-deps \
        /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
    python - <<'PY' || { echo "[ERROR] diff_gaussian_rasterization import failed"; return 2>/dev/null || exit 1; }
from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings  # noqa
print("[MIPGAUSSIAN] OK")
PY
fi

if [ "$VOX2SEQ" = true ] ; then
    echo "==> Building vox2seq from $WORKDIR/extensions/vox2seq"
    # vox2seq was removed from this fork (only serialized attention uses it, and
    # no shipped checkpoint does — see the header note). Restore the source from
    # the pre-removal commit before building, instead of failing on a cryptic cp.
    if [ ! -d "$WORKDIR/extensions/vox2seq" ] ; then
        echo "[INFO] extensions/vox2seq not in tree; restoring from commit 334d3b2"
        ( cd "$WORKDIR" && git archive 334d3b2 extensions/vox2seq | tar -x -C . ) \
            || { echo "[ERROR] could not restore extensions/vox2seq from git history"; return 2>/dev/null || exit 1; }
    fi
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/vox2seq
    cp -r "$WORKDIR/extensions/vox2seq" /tmp/extensions/vox2seq
    env CUDACXX="$CONDA_PREFIX/bin/nvcc" $PIP_BUILD_ENV \
        pip install -v /tmp/extensions/vox2seq --no-build-isolation
fi

if [ "$CUMM" = true ] ; then
    echo "==> Building cumm $CUMM_TAG from source (AOT, $CUDA_TAG, sm $TARGET_ARCH)"
    # WHY a separate --cumm step exists:
    #   * spconv 2.3.x requires cumm-cu128 in [0.7.11, 0.8.0). PyPI only ships
    #     cumm-cu128 0.8.2, so the prebuilt wheel cannot be used.
    #   * CUMM_DISABLE_JIT=1 is MANDATORY. Without it the wheel is Python-only
    #     (~370KB), no `cumm/core_cc*.so` is produced, and spconv's setup.py
    #     dies on `from cumm.core_cc import tensorview_bind`.
    #   * --no-deps so pip doesn't try to satisfy any transitive constraint by
    #     swapping out our pinned torch.
    pip uninstall -y cumm cumm-cu128 || true
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/cumm
    git clone --recursive https://github.com/FindDefinition/cumm.git /tmp/extensions/cumm
    pushd /tmp/extensions/cumm >/dev/null
    git checkout "$CUMM_TAG"
    git submodule update --init --recursive
    if [ ! -d "include/tensorview" ] || [ -z "$(ls -A include/tensorview 2>/dev/null)" ] ; then
        echo "[ERROR] cumm/include/tensorview is empty — submodule init failed."
        popd >/dev/null
        return 2>/dev/null || exit 1
    fi
    env CUDACXX="$CONDA_PREFIX/bin/nvcc" \
        CUMM_CUDA_VERSION="$CUDA_TOOLKIT_VER" \
        CUMM_CUDA_ARCH_LIST="$CUMM_ARCH" \
        CUMM_DISABLE_JIT=1 \
        TORCH_CUDA_ARCH_LIST="$TARGET_ARCH" \
        MAX_JOBS="${MAX_JOBS:-4}" \
        pip install -v --no-build-isolation --no-deps .
    popd >/dev/null
    # Fail loudly if the AOT extension wasn't produced.
    python - <<'PY' || { echo "[ERROR] cumm.core_cc missing — build went Python-only"; return 2>/dev/null || exit 1; }
import cumm
from cumm.core_cc import tensorview_bind
print(f"[CUMM] {cumm.__version__} OK (core_cc loaded)")
PY
fi

if [ "$SPCONV" = true ] ; then
    echo "==> Building spconv from source (no cu128 wheels with sm_100 yet)"
    # Precondition: cumm 0.7.x with core_cc must already be importable.
    python - <<'PY' || { echo "[ERROR] cumm not ready. Run setup_b200.sh --cumm first."; return 2>/dev/null || exit 1; }
import cumm
from cumm.core_cc import tensorview_bind  # noqa
v = cumm.__version__
assert v.startswith("0.7."), f"spconv 2.3.x needs cumm 0.7.x, got {v}"
PY
    pip uninstall -y spconv spconv-cu128 spconv-cu120 || true
    mkdir -p /tmp/extensions
    rm -rf /tmp/extensions/spconv
    git clone --recursive https://github.com/traveller59/spconv.git /tmp/extensions/spconv
    pushd /tmp/extensions/spconv >/dev/null
    # CUMM_CUDA_VERSION tells cumm/spconv which CUDA toolkit we're targeting.
    # SPCONV_DISABLE_JIT=1 forces AOT compilation of spconv/core_cc.
    # --no-deps so spconv's metadata phase can't bump torch.
    env CUDACXX="$CONDA_PREFIX/bin/nvcc" \
        CUMM_CUDA_VERSION="$CUDA_TOOLKIT_VER" \
        CUMM_CUDA_ARCH_LIST="$CUMM_ARCH" \
        SPCONV_DISABLE_JIT=1 \
        TORCH_CUDA_ARCH_LIST="$TARGET_ARCH" \
        MAX_JOBS="${MAX_JOBS:-4}" \
        pip install -v --no-build-isolation --no-deps .
    popd >/dev/null
    # Sanity: core_cc.so must exist; importing must succeed.
    python - <<'PY' || { echo "[ERROR] spconv.core_cc missing"; return 2>/dev/null || exit 1; }
import spconv, spconv.pytorch
print(f"[SPCONV] {spconv.__version__} OK")
PY
fi

if [ "$KAOLIN" = true ] ; then
    echo "==> Building kaolin"
    # Kaolin's S3 wheel index only has wheels for torch 2.7.0_cu128 (no 2.11.0
    # yet). For our torch 2.11 pin, the wheel path will miss and we'll source
    # build. --no-deps in both paths so kaolin can't bump our torch.
    pip uninstall -y kaolin || true
    KAOLIN_WHEEL_URL="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_VER}_${CUDA_TAG}.html"
    # --no-index is CRITICAL: without it, when the S3 index 404s (no wheel for
    # this torch/cu combo), pip silently falls back to PyPI and installs the
    # `kaolin` PLACEHOLDER package (a 917-byte wheel whose __init__ just raises
    # ImportError). That makes this branch "succeed" while leaving kaolin broken.
    # --no-index forces pip to consider ONLY the -f links, so a missing wheel
    # cleanly fails and we drop to the source build below.
    if pip install --no-deps --no-index kaolin -f "$KAOLIN_WHEEL_URL" 2>/dev/null ; then
        echo "[KAOLIN] Installed prebuilt wheel"
    else
        echo "[KAOLIN] No prebuilt wheel for torch ${TORCH_VER}/${CUDA_TAG}; building from source"
        mkdir -p /tmp/extensions
        rm -rf /tmp/extensions/kaolin
        git clone --recursive https://github.com/NVIDIAGameWorks/kaolin.git /tmp/extensions/kaolin
        pushd /tmp/extensions/kaolin >/dev/null
        # --no-build-isolation so the build sees our pinned torch (otherwise
        # pip's isolated build env pulls a fresh torch and the resulting .so
        # links against that, not ours → undefined-symbol crash at import).
        env $PIP_BUILD_ENV FORCE_CUDA=1 IGNORE_TORCH_VER=1 \
            pip install -v --no-deps --no-build-isolation .
        popd >/dev/null
    fi
    # Kaolin 0.18.x imports several runtime deps at package-init time that
    # --no-deps skips, so `import kaolin` dies before the C extension loads:
    #   wget, pygltflib            -> io
    #   warp-lang                  -> ops.pointcloud (NOT optional in 0.18)
    #   dataclasses-json,deprecated-> pygltflib's own deps (io.gltf)
    # Install them explicitly. warp-lang ships its own CUDA runtime (no torch
    # dep), and the dataclasses-json chain is pure-python — none touch our
    # torch pin. usd-core / ipyevents stay optional (only warn if absent).
    pip install --no-deps wget pygltflib warp-lang deprecated
    pip install dataclasses-json
    # Sanity: import kaolin's C extension end-to-end.
    python - <<'PY' || { echo "[ERROR] kaolin._C failed to import"; return 2>/dev/null || exit 1; }
import kaolin
from kaolin import _C
from kaolin.utils.testing import check_tensor  # noqa
print(f"[KAOLIN] {kaolin.__version__} OK (_C loaded)")
PY
fi

echo
echo "==> Done."
echo "Activate with: conda activate $ENV_NAME"
echo "Verify B200 + extensions:"
echo "  python -c 'import torch; print(torch.cuda.get_device_capability(0))'   # (10, 0)"
echo "  python -c 'import torchvision.ops as o; print(o.nms)'                  # function, not RuntimeError"
echo "  python -c 'import cumm; from cumm.core_cc import tensorview_bind; print(cumm.__version__)'  # 0.7.11"
echo "  python -c 'import spconv.pytorch; print(spconv.__version__)'           # 2.3.x"
echo "  python -c 'import nvdiffrast.torch; print(\"nvdiffrast OK\")'"
