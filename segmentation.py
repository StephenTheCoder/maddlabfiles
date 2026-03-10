import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# ── Labels for the semantic drone dataset ─────────────────────────────────
id2label = {
    0: "unlabeled", 1: "paved-area", 2: "dirt", 3: "grass", 4: "gravel",
    5: "water", 6: "rocks", 7: "pool", 8: "vegetation", 9: "roof",
    10: "wall", 11: "window", 12: "door", 13: "fence", 14: "fence-pole",
    15: "person", 16: "dog", 17: "car", 18: "bicycle", 19: "tree",
    20: "bald-tree", 21: "ar-marker", 22: "obstacle", 23: "conflicting"
}


# Only a few classes
# id2label = {
#     0: "unlabeled", 1: "paved-area", 2: "dirt", 3: "grass", 4: "tree",
#     5: "water", 6: "vegetation", 7: "conflicting"
# }
label2id = {v: k for k, v in id2label.items()}
num_classes = len(id2label)

# ── Load model ─────────────────────────────────────────────────────────────
model_name = "nickmuchi/segformer-b4-finetuned-segments-sidewalk"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(
    model_name,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Running on: {device}\n")

# Warmup
dummy = Image.new("RGB", (512, 512))
inputs = processor(images=dummy, return_tensors="pt").to(device)
with torch.no_grad():
    model(**inputs)

# ── Run segmentation across folder ────────────────────────────────────────
image_folder = "RandomImages"  # Change to your folder path
image_files = sorted([f for f in os.listdir(image_folder)
                      if f.lower().endswith((".jpg", ".jpeg", ".png"))])

latencies = []
unique_class_counts = []

for filename in image_files:
    path = os.path.join(image_folder, filename)
    img = Image.open(path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(**inputs)
    end = time.perf_counter()

    latency_ms = (end - start) * 1000
    latencies.append(latency_ms)

    # Get segmentation map
    seg_map = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[img.size[::-1]]
    )[0].cpu().numpy()

    # Count unique classes
    unique, counts = np.unique(seg_map, return_counts=True)
    n_unique = len(unique)
    unique_class_counts.append(n_unique)
    dominant_class = id2label[unique[np.argmax(counts)]]

    print(f"{filename:<30} | Classes: {n_unique:<3} | Dominant: {dominant_class:<15} | {latency_ms:.2f} ms")

    # ── Segmentation overlay ───────────────────────────────────────────────
    cmap = plt.get_cmap("tab20", num_classes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(filename, fontsize=12, fontweight="bold")

    ax1.imshow(img)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2.imshow(seg_map, cmap=cmap, vmin=0, vmax=num_classes - 1)
    ax2.set_title("Segmentation Map")
    ax2.axis("off")

    patches = [mpatches.Patch(color=cmap(i / num_classes), label=id2label[i])
               for i in unique]
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)

    plt.tight_layout()
    os.makedirs("outputs/random1", exist_ok=True)
    out_path = os.path.join("outputs/random", filename.rsplit(".", 1)[0] + "_seg.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()

# ── Overall latency stats ──────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Images processed: {len(latencies)}")
print(f"  Mean latency:     {np.mean(latencies):.2f} ms")
print(f"  Median latency:   {np.median(latencies):.2f} ms")
print(f"  Min latency:      {np.min(latencies):.2f} ms")
print(f"  Max latency:      {np.max(latencies):.2f} ms")
print(f"  Mean FPS:         {1000/np.mean(latencies):.1f}")
print(f"{'='*50}")

# ── Plot 1: Latency per image ──────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle("Segmentation Performance", fontsize=14, fontweight="bold")

ax1.bar(range(len(latencies)), latencies, color="mediumpurple", edgecolor="white")
ax1.axhline(np.mean(latencies), color="red", linestyle="--",
            label=f"Mean: {np.mean(latencies):.1f} ms")
ax1.set_xlabel("Image Index")
ax1.set_ylabel("Latency (ms)")
ax1.set_title("Latency per Image")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# ── Plot 2: Latency vs number of unique classes ────────────────────────────
# Bin by class count and compute averages for the trend line
unique_counts_sorted = sorted(set(unique_class_counts))
avg_latency_per_count = [
    np.mean([latencies[i] for i, c in enumerate(unique_class_counts) if c == n])
    for n in unique_counts_sorted
]

ax2.scatter(unique_class_counts, latencies, alpha=0.5, color="steelblue", label="Per image")
ax2.plot(unique_counts_sorted, avg_latency_per_count, color="red", linewidth=2,
         marker="o", label="Avg latency")
ax2.set_xlabel("Number of Unique Classes Detected")
ax2.set_ylabel("Latency (ms)")
ax2.set_title("Latency vs Number of Unique Classes Detected")
ax2.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax2.legend()
ax2.grid(alpha=0.3)

stats = (f"Mean: {np.mean(latencies):.1f} ms  |  Median: {np.median(latencies):.1f} ms  |  "
         f"Min: {np.min(latencies):.1f} ms  |  Max: {np.max(latencies):.1f} ms  |  "
         f"FPS: {1000/np.mean(latencies):.1f}")
fig.text(0.5, 0.01, stats, ha="center", fontsize=9, color="gray")

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig("outputs/random1/performance_plots_random.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved to outputs/random1/performance_plots_random.png")