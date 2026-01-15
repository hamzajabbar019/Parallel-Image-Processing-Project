import os
import time
import argparse
from PIL import Image, ImageFilter, ImageEnhance
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt

# -----------------------------
# Apply one filter to image
# -----------------------------
def apply_filter(img, filter_name, blur=2.0, brightness=1.0, contrast=1.0):
    if filter_name == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=blur))
    if filter_name == "edges":
        return img.filter(ImageFilter.FIND_EDGES)
    if filter_name == "bright":
        out = img
        if brightness != 1.0:
            out = ImageEnhance.Brightness(out).enhance(brightness)
        if contrast != 1.0:
            out = ImageEnhance.Contrast(out).enhance(contrast)
        return out
    raise ValueError(f"Unknown filter: {filter_name}")

# -----------------------------
# Split image into horizontal chunks
# -----------------------------
def build_chunks(img_height, chunk_height, overlap):
    chunks = []
    y = 0
    while y < img_height:
        y0 = y
        y1 = min(y + chunk_height, img_height)
        crop_top = max(0, y0 - overlap)
        crop_bottom = min(img_height, y1 + overlap)
        chunks.append((y0, y1, crop_top, crop_bottom))
        y = y1
    return chunks

# -----------------------------
# Process one chunk
# -----------------------------
def process_chunk(img, chunk, filter1, filter2, blur, brightness, contrast, overlap):
    y0, y1, crop_top, crop_bottom = chunk
    strip = img.crop((0, crop_top, img.width, crop_bottom))
    
    strip = apply_filter(strip, filter1, blur, brightness, contrast)
    if filter2:
        strip = apply_filter(strip, filter2, blur, brightness, contrast)
    
    # Remove overlap
    trim_top = y0 - crop_top
    trim_bottom = crop_bottom - y1
    w, h = strip.size
    strip = strip.crop((0, trim_top, w, h - trim_bottom))
    return y0, strip

# -----------------------------
# Sequential processing
# -----------------------------
def process_sequential(img, filter1, filter2, blur, brightness, contrast):
    out = apply_filter(img, filter1, blur, brightness, contrast)
    if filter2:
        out = apply_filter(out, filter2, blur, brightness, contrast)
    return out

# -----------------------------
# Parallel processing
# -----------------------------
def process_parallel(img, filter1, filter2, workers=4, chunk_h=256, overlap=16,
                     blur=2.0, brightness=1.0, contrast=1.0):
    out = Image.new(img.mode, img.size)
    chunks = build_chunks(img.height, chunk_h, overlap)
    
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_chunk, img, c, filter1, filter2, blur, brightness, contrast, overlap) for c in chunks]
        for f in futures:
            y0, strip = f.result()
            out.paste(strip, (0, y0))
    return out

# -----------------------------
# Run image (seq or par)
# -----------------------------
def run_image(mode, img, filter1, filter2, blur, brightness, contrast, workers, chunk_h, overlap):
    if mode == "seq":
        return process_sequential(img, filter1, filter2, blur, brightness, contrast)
    if mode == "par":
        return process_parallel(img, filter1, filter2, workers, chunk_h, overlap, blur, brightness, contrast)
    raise ValueError(f"Unknown mode: {mode}")

# -----------------------------
# Benchmark with automatic graphs
# -----------------------------
def benchmark_images(image_paths, filter1, filter2, workers=4, repeats=5,
                     blur=2.0, brightness=1.0, contrast=1.0,
                     chunk_h=256, overlap=16, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    
    images = []
    seq_times = []
    par_times = []
    speedups = []

    for path in image_paths:
        name = os.path.basename(path)
        img = Image.open(path).convert("RGB")

        # Warmup
        _ = run_image("seq", img, filter1, filter2, blur, brightness, contrast, workers, chunk_h, overlap)
        _ = run_image("par", img, filter1, filter2, blur, brightness, contrast, workers, chunk_h, overlap)

        # Sequential timing
        t_seq = sum([time.perf_counter() - time.perf_counter() + 0 for _ in range(repeats)])  # dummy to avoid zero
        t_seq = sum([time.perf_counter() for _ in range(repeats)])  # proper timing
        t_seq = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = run_image("seq", img, filter1, filter2, blur, brightness, contrast, workers, chunk_h, overlap)
            t1 = time.perf_counter()
            t_seq += (t1 - t0)
        t_seq /= repeats

        # Parallel timing
        t_par = 0
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = run_image("par", img, filter1, filter2, blur, brightness, contrast, workers, chunk_h, overlap)
            t1 = time.perf_counter()
            t_par += (t1 - t0)
        t_par /= repeats

        speedup = t_seq / t_par if t_par > 0 else 0

        # Save processed images for report
        out_seq = run_image("seq", img, filter1, filter2, blur, brightness, contrast, workers, chunk_h, overlap)
        out_seq.save(os.path.join(out_dir, f"{name}_seq.jpg"))
        out_par = run_image("par", img, filter1, filter2, blur, brightness, contrast, workers, chunk_h, overlap)
        out_par.save(os.path.join(out_dir, f"{name}_par.jpg"))

        # Append for table and graphs
        images.append(name)
        seq_times.append(round(t_seq, 3))
        par_times.append(round(t_par, 3))
        speedups.append(round(speedup, 2))

        print(f"{name}: Par={t_seq:.3f}s, Seq={t_par:.3f}s, Speedup={speedup:.2f}x")

    # Plot graphs
    plt.figure(figsize=(8,5))
    plt.bar(images, seq_times, width=0.4, label='Parallel', align='edge')
    plt.bar(images, par_times, width=-0.4, label='Sequential', align='edge')
    plt.ylabel("Time (s)")
    plt.title("Sequential vs Parallel Processing Time")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "time_comparison.png"))
    plt.show()

    plt.figure(figsize=(8,5))
    plt.bar(images, speedups, color='green')
    plt.ylabel("Speedup (x)")
    plt.title("Parallel Speedup")
    plt.savefig(os.path.join(out_dir, "speedup.png"))
    plt.show()

    return images, seq_times, par_times, speedups

# -----------------------------
# CLI Commands
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Parallel Image Processing")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Single image
    s = sub.add_parser("single")
    s.add_argument("--input", required=True)
    s.add_argument("--output", required=True)
    s.add_argument("--mode", choices=["seq", "par"], default="seq")
    s.add_argument("--filter1", choices=["blur", "edges", "bright"], required=True)
    s.add_argument("--filter2", choices=["blur", "edges", "bright"], default=None)
    s.add_argument("--blur", type=float, default=2.0)
    s.add_argument("--brightness", type=float, default=1.0)
    s.add_argument("--contrast", type=float, default=1.0)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--chunk-h", type=int, default=256)
    s.add_argument("--overlap", type=int, default=16)

    # Batch
    b = sub.add_parser("batch")
    b.add_argument("--in-dir", required=True)
    b.add_argument("--out-dir", required=True)
    b.add_argument("--mode", choices=["seq", "par"], default="par")
    b.add_argument("--filter1", choices=["blur", "edges", "bright"], required=True)
    b.add_argument("--filter2", choices=["blur", "edges", "bright"], default=None)
    b.add_argument("--blur", type=float, default=2.0)
    b.add_argument("--brightness", type=float, default=1.0)
    b.add_argument("--contrast", type=float, default=1.0)
    b.add_argument("--workers", type=int, default=4)
    b.add_argument("--chunk-h", type=int, default=256)
    b.add_argument("--overlap", type=int, default=16)

    # Benchmark
    bench = sub.add_parser("bench")
    bench.add_argument("--in-dir", required=True)
    bench.add_argument("--filter1", choices=["blur", "edges", "bright"], required=True)
    bench.add_argument("--filter2", choices=["blur", "edges", "bright"], default=None)
    bench.add_argument("--blur", type=float, default=2.0)
    bench.add_argument("--brightness", type=float, default=1.0)
    bench.add_argument("--contrast", type=float, default=1.0)
    bench.add_argument("--workers", type=int, default=4)
    bench.add_argument("--chunk-h", type=int, default=256)
    bench.add_argument("--overlap", type=int, default=16)
    bench.add_argument("--repeats", type=int, default=5)
    bench.add_argument("--out-dir", default="../output")

    args = parser.parse_args()

    if args.cmd == "single":
        img = Image.open(args.input).convert("RGB")
        out = run_image(args.mode, img, args.filter1, args.filter2,
                        args.blur, args.brightness, args.contrast,
                        args.workers, args.chunk_h, args.overlap)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        out.save(args.output)
        print(f"Saved: {args.output}")

    elif args.cmd == "batch":
        os.makedirs(args.out_dir, exist_ok=True)
        files = [f for f in os.listdir(args.in_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        for f in files:
            in_path = os.path.join(args.in_dir, f)
            out_path = os.path.join(args.out_dir, f)
            img = Image.open(in_path).convert("RGB")
            out = run_image(args.mode, img, args.filter1, args.filter2,
                            args.blur, args.brightness, args.contrast,
                            args.workers, args.chunk_h, args.overlap)
            out.save(out_path)
            print(f"Processed: {f}")

    elif args.cmd == "bench":
        files = [os.path.join(args.in_dir, f) for f in os.listdir(args.in_dir)
                 if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        benchmark_images(files, args.filter1, args.filter2,
                         workers=args.workers, repeats=args.repeats,
                         blur=args.blur, brightness=args.brightness,
                         contrast=args.contrast, chunk_h=args.chunk_h,
                         overlap=args.overlap, out_dir=args.out_dir)

if __name__ == "__main__":
    main()
