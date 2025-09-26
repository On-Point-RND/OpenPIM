import os
import sys
from PIL import Image, ImageDraw, ImageFont


FILTER_E = ["cond", "ext", "total"]
FILTER_D = []  # Adjust if needed to match 3 dataset names


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

    # dataset_dirs = [d for d in dataset_dirs if any(f in d for f in FILTER_D)]

    if len(dataset_dirs) != 3:
        print(f"Expected 3 datasets, found {len(dataset_dirs)}")
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

    experiment_dirs = [e for e in experiment_dirs if any(f in e for f in FILTER_E)]

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
                target_size = img.size
            else:
                img = img.resize(target_size, Image.BILINEAR)
            images.append(img)

    # Step 4: Create a new image canvas with 3 columns and 3 rows + 1 header row
    w, h = target_size
    total_width = 3 * w
    header_height = h // 10  # Adjust as needed
    total_height = 3 * h + header_height
    final_image = Image.new(
        "RGB", (total_width, total_height), color=(255, 255, 255)
    )  # White background
    draw = ImageDraw.Draw(final_image)

    # Draw dataset names at the top of each column
    try:
        font = ImageFont.truetype("arial.ttf", size=24)  # Use a larger font
    except IOError:
        font = ImageFont.load_default()  # Fallback to default
        # Manually scale up text size (approximate)
        font_size = 24
    font = ImageFont.load_default().font_variant(size=font_size)
    # Draw dataset names at the top of each column
    for col_idx, dataset in enumerate(dataset_dirs):
        text = dataset
        _, _, text_width, text_height = draw.textbbox((0, 0), text=text, font=font)

        x = col_idx * w + (w - text_width) / 2
        y = (header_height - text_height) / 2  # Center in smaller header
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

    # Step 5: Paste images into grid (row-major order: 3 datasets per row)
    for i, img in enumerate(images):
        row = i // 3
        col = i % 3
        position = (col * w, row * h + header_height)
        final_image.paste(img, position)

    # Step 6: Save the final combined image
    output_path = os.path.join(root, "combined_performance_grid.png")
    final_image.save(output_path)
    print(f"Combined image saved at: {output_path}")


if __name__ == "__main__":
    main()
