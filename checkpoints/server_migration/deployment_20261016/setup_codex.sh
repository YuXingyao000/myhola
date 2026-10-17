#!/usr/bin/env bash
set -euo pipefail

# Run on the destination host. Source credentials are never embedded here.
node_version=v24.22.0
codex_version=0.160.1
tools_root=/mnt/d/tools
node_name="node-${node_version}-linux-x64"
download_root=/mnt/d/migration/downloads
mkdir -p "$tools_root" "$download_root" /mnt/d/migration/logs

if [[ ! -x "$tools_root/$node_name/bin/node" ]]; then
    curl --fail --location --retry 3 \
        "https://nodejs.org/dist/${node_version}/${node_name}.tar.xz" \
        --output "$download_root/${node_name}.tar.xz"
    curl --fail --location --retry 3 \
        "https://nodejs.org/dist/${node_version}/SHASUMS256.txt" \
        --output "$download_root/node-SHASUMS256.txt"
    (
        cd "$download_root"
        awk -v name="${node_name}.tar.xz" '$2 == name {print}' node-SHASUMS256.txt > node-selected.sha256
        test -s node-selected.sha256
        sha256sum --check node-selected.sha256
    )
    tar -xJf "$download_root/${node_name}.tar.xz" -C "$tools_root"
fi

ln -sfn "$tools_root/$node_name" "$tools_root/node"
for executable in node npm npx; do
    ln -sfn "$tools_root/node/bin/$executable" "/usr/local/bin/$executable"
done
export PATH="$tools_root/node/bin:$PATH"
npm install --global "@openai/codex@${codex_version}"
ln -sfn "$tools_root/node/bin/codex" /usr/local/bin/codex
node --version
npm --version
codex --version
codex --help >/mnt/d/migration/logs/codex-help.txt
