# ocr_pipeline/pdf_loader.py
from __future__ import annotations

import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List
from pdf2image import convert_from_path, pdfinfo_from_path

def _convert_page_range(
    pdf_path: str,
    start_page: int,
    end_page: int,
    dpi: int,
    thread_count: int
) -> List:
    """
    Convert a specific range of PDF pages to images.

    Args:
        pdf_path (str): Path to the PDF file
        start_page (int): Starting page number (1-indexed)
        end_page (int): Ending page number (1-indexed)
        dpi (int): DPI for conversion
        thread_count (int): Number of threads for pdf2image

    Returns:
        List: List of PIL images for the page range
    """
    return convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=start_page,
        last_page=end_page,
        thread_count=thread_count
    )

def _save_image(args):
    """
    Save a single image to disk.

    Args:
        args: Tuple of (page_image, img_path, format)
    """
    page, img_path, format = args
    page.save(img_path, format=format.upper())
    return img_path

def pdf_to_images(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
    format: str = "png",
    num_processes: int | None = None,
    num_threads: int | None = None
) -> List[Path]:
    """
    Convert a PDF into a list of image files (one per page) using parallelism.

    Args:
        pdf_path (Path): Path to the input PDF file.
        out_dir (Path): Directory to save the output images.
        dpi (int, optional): Dots per inch for image quality. Defaults to 200.
        format (str, optional): Image format (e.g., 'png', 'jpeg'). Defaults to 'png'.
        num_processes (int, optional): Number of processes for parallel page conversion. 
                                      Defaults to CPU count // 2.
        num_threads (int, optional): Number of threads for parallel image saving.
                                    Defaults to 4.

    Returns:
        List[Path]: List of paths to the generated image files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() // 2)
    if num_threads is None:
        num_threads = 4
    
    # First, get total number of pages
    info = pdfinfo_from_path(str(pdf_path))
    total_pages = info["Pages"]
    
    if total_pages == 0:
        return []
    
    # Divide pages into chunks for parallel processing
    pages_per_process = max(1, total_pages // num_processes)
    page_ranges = []
    
    start_page = 1
    for _ in range(num_processes):
        end_page = min(start_page + pages_per_process - 1, total_pages) # inclusive
        if start_page <= end_page:
            page_ranges.append((start_page, end_page))
        start_page = end_page + 1 # move to next range
        if start_page > total_pages:
            break
    
    # Convert page ranges in parallel using multiprocessing
    with multiprocessing.Pool(processes=len(page_ranges)) as pool:
        results = pool.starmap(
            _convert_page_range,
            [(str(pdf_path), start, end, dpi, 2) for start, end in page_ranges]
        )

    # Flatten the results
    all_pages = []
    for result in results:
        all_pages.extend(result)

    # Prepare save tasks
    save_tasks = []
    for idx, page in enumerate(all_pages, start=1):
        img_name = f"page-{idx:03d}.{format}"
        img_path = out_dir / img_name
        save_tasks.append((page, img_path, format))

    # Save images in parallel using threading
    image_paths = []
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_task = {
            executor.submit(_save_image, task): task for task in save_tasks
        }

        # Collect results as they complete
        for future in as_completed(future_to_task):
            img_path = future.result()
            image_paths.append(img_path)

    # Sort by page number to maintain order
    image_paths.sort(key=lambda x: int(x.stem.split('-')[1]))

    return image_paths
