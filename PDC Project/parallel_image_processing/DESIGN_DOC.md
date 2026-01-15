Parallel Image Processing Design Document
Project Overview

This project implements image filtering techniques in Python using both sequential and parallel execution methods. By splitting the image into smaller chunks and processing them in parallel, we aim to significantly reduce the processing time, especially for large images.

Objective

To apply various image filters (blur, edge detection, brightness, contrast) to images.

To compare the performance of sequential versus parallel processing techniques.

To visualize the results and performance improvements.

System Design
Flow Diagram:

Input Image: The image is loaded into memory.

Split Image into Chunks: The image is split into horizontal chunks, each processed independently.

Apply Filters: The selected filters are applied to each chunk.

Merge Chunks: The processed chunks are merged to form the final output.

Output Image: The processed image is saved to disk.

Chunking with Overlap:

To avoid visible seams between the processed chunks, we introduce an overlap where adjacent chunks share a portion of the image.

The chunk size (chunk_h) and overlap (overlap) are configurable parameters.

Parallelism:

ThreadPoolExecutor from the concurrent.futures module is used to process chunks in parallel. Each chunk is handled by a separate thread.

Filters:

Blur Filter: Smooths the image and reduces noise.

Edge Detection: Detects the boundaries of objects in the image.

Brightness and Contrast: Adjusts the image's lighting and contrast to enhance visibility.

Code Structure
Filter Application Function

The apply_filter function applies the selected filter to an image chunk:

def apply_filter(img, filter_name, blur=2.0, brightness=1.0, contrast=1.0):
    if filter_name == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=blur))
    if filter_name == "edges":
        return img.filter(ImageFilter.FIND_EDGES)
    if filter_name == "bright":
        out = ImageEnhance.Brightness(img).enhance(brightness)
        out = ImageEnhance.Contrast(out).enhance(contrast)
        return out

Sequential Processing

The entire image is processed sequentially using a single thread:

def process_sequential(img, filter1, filter2, blur, brightness, contrast):
    out = apply_filter(img, filter1, blur, brightness, contrast)
    if filter2:
        out = apply_filter(out, filter2, blur, brightness, contrast)
    return out

Parallel Processing

The image is split into chunks, and each chunk is processed in parallel:

def process_parallel(img, filter1, filter2, workers=4, chunk_h=256, overlap=16, blur=2.0, brightness=1.0, contrast=1.0):
    out = Image.new(img.mode, img.size)
    chunks = build_chunks(img.height, chunk_h, overlap)
    
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_chunk, img, c, filter1, filter2, blur, brightness, contrast, overlap) for c in chunks]
        for f in futures:
            y0, strip = f.result()
            out.paste(strip, (0, y0))
    return out

Performance Evaluation
Speedup Formula:

The speedup of parallel processing over sequential processing is calculated as:

Speedup = Sequential Time / Parallel Time

Benchmarking:

The project includes a benchmarking feature that runs both sequential and parallel processes over multiple images and calculates the speedup.

Output:

Two output images for each input image:

Sequential processed image img1_seq.jpg)

Parallel processed image img1_par.jpg)

Performance graphs: speedup.png and time_comparison.png.