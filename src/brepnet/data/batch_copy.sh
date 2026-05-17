SRC=/mnt/d/data/abc_v2_new_imgs
DST=/mnt/d/data/deepcad_v6_cond

for dir in $SRC/*; do
    id=$(basename "$dir")

    src_file="$dir/new_imgs.npz"
    dst_dir="$DST/$id"

    if [ -f "$src_file" ] && [ -d "$dst_dir" ]; then
        cp "$src_file" "$dst_dir/sketch_and_natural.npz"
        echo "Added sketch_and_natural.npz → $id"
    else
        echo "Skip $id (missing src or dst)"
    fi
done