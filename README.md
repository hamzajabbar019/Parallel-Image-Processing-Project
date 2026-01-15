Parallel Image Processing using Sequential and Parallel Execution

Group members names:
Hamza Jabbar (2212251) 
Muhammad Osama (2212259) 
Sadam Hussain (2212266) 
Hasnain Magsi (2212252) 
Abdul Rehman (2112258) 
Danish (2012338)

Overview
This project demonstrates the application of image processing using both sequential and parallel execution techniques. The goal is to compare the execution time between the two methods by applying common filters (blur, edge detection, and brightness/contrast) to large images. Parallel processing reduces processing time by dividing the image into chunks and processing them concurrently using threads.

Features

Filters Available:

Blur: Applies a Gaussian Blur to smooth the image.

Edge Detection: Highlights the boundaries and edges of the image.

Brightness & Contrast: Adjusts the brightness and contrast of the image.

Processing Modes:

Sequential: The entire image is processed using a single thread.

Parallel: The image is divided into chunks, and each chunk is processed using multiple threads.

Benchmarking: The project also includes benchmarking functionality to measure and compare execution times for both sequential and parallel processing, providing insights into performance improvements.

Folder Structure
parallel_image_processing_project/
├── dataset/
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── img4.jpg
├── output/
│   ├── img1.jpg_par.jpg
│   ├── img1.jpg_seq.jpg
│   ├── img2.jpg_par.jpg
│   ├── img2.jpg_seq.jpg
│   ├── img3.jpg_par.jpg
│   ├── img3.jpg_seq.jpg
│   ├── img4.jpg_par.jpg
│   ├── img4.jpg_seq.jpg
│   ├── speedup.png
│   └── time_comparison.png
├── parallel_image_processing/
│   ├── DESIGN_DOC.md
│   ├── FINAL_REPORT.md
│   ├── image_processor.py
│   ├── README.md
│   ├── requirements.txt
└── README.md

Input Folder (dataset/)

This folder contains the images to be processed:

img1.jpg, img2.jpg, img3.jpg, img4.jpg

Output Folder (output/)

After processing, this folder contains:

Sequential and Parallel processed images for each input image (e.g., img1.jpg_seq.jpg and img1.jpg_par.jpg).

Performance comparison graphs: speedup.png and time_comparison.png.

Script Folder (parallel_image_processing/)

This folder contains the main processing script and documents:

image_processor.py: The main Python script that applies filters and processes images sequentially or in parallel.

requirements.txt: Lists required libraries.

DESIGN_DOC.md: Describes the design and system architecture.

FINAL_REPORT.md: Summarizes the project results and conclusions.

Installation

Install the required libraries:

pip install -r parallel_image_processing/requirements.txt

How to Use
Single Image Processing:

To process a single image:

python parallel_image_processing/image_processor.py single --input <input_image> --output <output_image> --mode <seq|par> --filter1 <filter_name> --filter2 <filter_name> --blur <value> --brightness <value> --contrast <value> --workers <num_workers>

Batch Processing:

To process all images in the dataset/ folder:

python parallel_image_processing/image_processor.py batch --in-dir dataset --out-dir output --mode <seq|par> --filter1 <filter_name> --filter2 <filter_name> --blur <value> --brightness <value> --contrast <value> --workers <num_workers>

Benchmarking:

To benchmark the performance of sequential vs parallel processing:

python parallel_image_processing/image_processor.py bench --in-dir dataset --filter1 <filter_name> --filter2 <filter_name> --blur <value> --brightness <value> --contrast <value> --workers <num_workers> --chunk-h <chunk_height> --overlap <overlap_value> --repeats <num_repeats> --out-dir output

Benchmark Results

The speedup.png graph shows the performance improvement when using parallel processing over sequential processing. The time_comparison.png graph compares the execution time of both methods for each image.

License

MIT License
