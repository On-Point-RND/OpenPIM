import os
import sys
from PIL import Image


def main():
    if len(sys.argv) != 2:
        print("Usage: python concatenate_images.py <root_directory>")
        return

    root = sys.argv[1]

    if not os.path.isdir(root):
        print(f"Error: Root directory '{root}' does not exist or is not a directory.")
        return

    # Step 1: Find and sort datasets
    dataset_dirs = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    )

    if len(dataset_dirs) != 4:
        print(f"Expected 4 datasets, found {len(dataset_dirs)}")
        return

    # Step 2: Find and sort experiments (from first dataset)
    first_dataset_path = os.path.join(root, dataset_dirs[0])
    experiment_dirs = sorted(
        [
            e
            for e in os.listdir(first_dataset_path)
            if os.path.isdir(os.path.join(first_dataset_path, e))
        ]
    )

    if len(experiment_dirs) != 3:
        print(f"Expected 3 experiments, found {len(experiment_dirs)}")
        return

    # Step 3: Load and resize all images to match the first image's size
    images = []
    target_size = None
    for exp in experiment_dirs:
        for ds in dataset_dirs:
            img_path = os.path.join(root, ds, exp, "barplot_perfofmance.png")
            if not os.path.exists(img_path):
                print(f"Missing image file: {img_path}")
                return
            img = Image.open(img_path)
            if target_size is None:
                target_size = img.size  # Use first image's size as reference
            else:
                img = img.resize(target_size, Image.BILINEAR)  # Resize others to match
            images.append(img)

    # Step 4: Create a new image canvas with 4 columns and 3 rows
    w, h = target_size
    total_width = 4 * w
    total_height = 3 * h
    final_image = Image.new("RGB", (total_width, total_height))

    # Step 5: Paste images into grid (row-major order: 4 datasets per row)
    for i, img in enumerate(images):
        row = i // 4
        col = i % 4
        position = (col * w, row * h)
        final_image.paste(img, position)

    # Step 6: Save the final combined image
    output_path = os.path.join(root, "combined_performance_grid.png")
    final_image.save(output_path)
    print(f"Combined image saved at: {output_path}")


if __name__ == "__main__":
    main()
