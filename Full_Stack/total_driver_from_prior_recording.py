import subprocess, time, json, urllib.request, socket, contextlib, signal, asyncio
from pathlib import Path
from pyppeteer import launch           # pip install pyppeteer
import os
from PyPDF2 import PdfMerger

# ── 1.  pick a free TCP port every run ───────────────────────────
def pick_free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

# ── 2.  kill-and-dump helper (never blocks) ─────────────────────
def dump_log_and_raise(msg: str, proc: subprocess.Popen):
    """Terminate child, print whatever it wrote so far, then raise."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    # read *after* the process is gone → never blocks
    out, _ = proc.communicate()
    print("\n--- Streamlit log ---")
    print(out.decode(errors="replace") if isinstance(out, bytes) else out)
    print("--- end log ---\n")

    raise RuntimeError(msg)

# ── 3.  wait for the /health endpoint (no blocking reads) ───────
HEALTH_PATHS = ["/healthz", "/_stcore/health"]

# ----- keep both old & new formats -------------------------------
def _is_ok(payload: bytes) -> bool:
    try:                        # new: just the word "ok"
        return payload.strip().lower() == b"ok"
    except Exception:
        return False

def wait_until_up(port: int, proc: subprocess.Popen, timeout=20):
    url_tpls = [f"http://localhost:{port}{p}" for p in ("/healthz", "/_stcore/health")]
    deadline = time.time() + timeout

    while time.time() < deadline:
        if proc.poll() is not None:
            dump_log_and_raise("Streamlit exited prematurely", proc)

        for url in url_tpls:
            try:
                data = urllib.request.urlopen(url, timeout=1).read()
                if _is_ok(data) or json.loads(data).get("status") == "ok":
                    return
            except Exception:
                pass
        time.sleep(0.3)

    dump_log_and_raise("Streamlit never became healthy", proc)

async def page_to_pdf(url: str, pdf_path: Path):
    browser = await launch(headless=True, args=["--no-sandbox"])
    page    = await browser.newPage()
    await page.goto(url, waitUntil="networkidle2")

    # wait until Streamlit inserts the sentinel
    await page.waitForSelector("#done", timeout=120_000)   # 120 s max

    await page.pdf(path=str(pdf_path), printBackground=True)
    await browser.close()

# ── 5.  main helper: never blocks on stdout, always cleans up ───
def run_motion_processing(python_path, motion_processing_script, data_npz_file, pdf_out, port=None):
    port = port or pick_free_port()

    cmd = [
        python_path, "-m", "streamlit", "run", motion_processing_script,
        "--server.headless", "true",
        "--server.port", str(port),
        "--", "--data_path", data_npz_file,
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1
    )

    try:
        wait_until_up(port, proc)                               # health probe
        asyncio.run(page_to_pdf(f"http://localhost:{port}",     # write PDF
                                Path(pdf_out)))

        # ── we’re done with the app; ask it to shut down ──────
        if proc.poll() is None:
            proc.terminate()              # polite SIGTERM
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()               # hard kill if needed

    finally:
        if proc.poll() is None:           # safety net
            proc.kill()



    base = os.path.basename(data_npz_file).replace('.npz', '')
    base = base.replace('pixel_trajectories_', '')

    return os.path.join(os.path.dirname(data_npz_file), f'rgb_per_region_{base}.npy')



# def run_video_acquisition(python_path, video_acquisiton_script):

#     video_name = input('Enter video save name (without extension): ')

#     subprocess.run([
#         python_path,
#         video_acquisiton_script,
#         '--video_name', video_name,
#     ])


#     print('\n')
#     print('\n')
#     print('################################################')
#     print('################################################')
#     print('\n')
#     print("Video acquisition completed.")
#     print('\n')
#     print('################################################')
#     print('################################################')
#     print('\n')
#     print('\n')

#     return os.path.join('/Users/henryschnieders/Documents/Research/My_Data', video_name + '_wholeface.npy')


def run_pixel_matching(python_path, pixel_matching_script, data_npy_file):

    subprocess.run([
        python_path, 
        pixel_matching_script,
        '--data_path', data_npy_file,
    ])



    print('\n')
    print('\n')
    print('################################################')
    print('################################################')
    print('\n')
    print("Pixel matching completed.")
    print('\n')
    print('################################################')
    print('################################################')
    print('\n')
    print('\n')
    

    base = os.path.basename(data_npy_file).replace('.npy', '')

    return f"/Users/henryschnieders/Documents/Research/My_Data/pixel_trajectories_{base}.npz"

    

def run_CHROM_method_wmotionprocess(python_path, CHROM_method_script_with, motion_processed_paths):

    result = subprocess.run([
                python_path,
                CHROM_method_script_with,
                '--motion_processed_paths', motion_processed_paths],
                capture_output=True, text=True, check = True
                )

    print('\n')
    print('\n')
    print('################################################')
    print('################################################')
    print('\n')
    print("CHROM method completed.")
    print('\n')
    print('################################################')
    print('################################################')
    print('\n')
    print('\n')

    return result.stdout.strip() if result.stdout else None


def run_CHROM_method_nomotionprocess(python_path, CHROM_method_script_without, data_npz_file):

    result = subprocess.run([
                python_path,
                CHROM_method_script_without,
                '--data_path', data_npz_file],
                capture_output=True, text=True, check = True
                )       


    print('\n')
    print('\n')
    print('################################################')
    print('################################################')
    print('\n')
    print("CHROM method completed.")
    print('\n')
    print('################################################')
    print('################################################')
    print('\n')
    print('\n')

    return result.stdout.strip() if result.stdout else None




if __name__ == '__main__':

    python_path = '/opt/homebrew/opt/python@3.12/bin/python3.12'

    video_acquisition_script = '/Users/henryschnieders/Documents/Research/My_work_parts/Video_get_to_npy/VID_TO_DATA_whole_face.py'
    pixel_matching_script = '/Users/henryschnieders/Documents/Research/My_work_parts/motion_process/temporal_pixel_matching/forward_backward_pixel_matching.py'
    motion_processing_script = '/Users/henryschnieders/Documents/Research/My_work_parts/motion_process/occulsion_processing/motion_process_fb.py'
    CHROM_method_script_with = '/Users/henryschnieders/Documents/Research/My_work_parts/chrom_method/CHROM_method_npy_input.py'
    CHROM_method_script_without = '/Users/henryschnieders/Documents/Research/My_work_parts/chrom_method/CHROM_method_npz_input.py'


    data_npy_file = '/path/to/video/processed_by/.../VID_TO_DATA_whole_face.py'

    data_npz_file = run_pixel_matching(
                        python_path           = python_path,
                        pixel_matching_script = pixel_matching_script,
                        data_npy_file         = data_npy_file,
                        )
    
    

    # run as streamlit: /opt/homebrew/bin/streamlit
    motion_plots_path = os.path.splitext(os.path.basename(data_npy_file))[0] + ".pdf"
    motion_plots_path = os.path.join('/Users/henryschnieders/Documents/Research/My_Data', motion_plots_path)

    motion_processed_paths  = run_motion_processing(
                                    python_path              = python_path,
                                    motion_processing_script = motion_processing_script,
                                    data_npz_file            = data_npz_file,
                                    pdf_out                  = motion_plots_path
                                    )

    chrom_pdf_1 = run_CHROM_method_wmotionprocess(
                        python_path               = python_path,
                        CHROM_method_script_with  = CHROM_method_script_with,
                        motion_processed_paths    = motion_processed_paths,
                    )

    chrom_pdf_2 = run_CHROM_method_nomotionprocess(
                        python_path                  = python_path,
                        CHROM_method_script_without  = CHROM_method_script_without,
                        data_npz_file                = data_npz_file,
                    )


    merger = PdfMerger()
    for pdf in (motion_plots_path, chrom_pdf_1, chrom_pdf_2):
        merger.append(pdf)

    final_pdf = motion_plots_path.replace(".pdf", "_FULL.pdf")
    with open(final_pdf, "wb") as f_out:
        merger.write(f_out)

    print("✅  Combined PDF written to", final_pdf)