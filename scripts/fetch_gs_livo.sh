#!/bin/sh
# Clone outside the catkin workspace; never overwrite a user's checkout.
set -eu
dst=${1:?usage: fetch_gs_livo.sh NEW_DIRECTORY_OUTSIDE_CATKIN}
pin=cc65db279bf0d7c4df12d21710cc4053753dd133
test ! -e "$dst" || { echo "destination exists: $dst" >&2; exit 1; }
git clone --no-checkout --filter=blob:none --sparse https://github.com/HKUST-Aerial-Robotics/GS-LIVO.git "$dst"
git -C "$dst" sparse-checkout set src/lib3dgs src/glm
git -C "$dst" checkout --detach "$pin"
test "$(git -C "$dst" rev-parse HEAD)" = "$pin"
echo "Set GS_LIVO_ROOT to $dst"
