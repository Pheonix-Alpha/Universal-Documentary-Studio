"""
colab_worker.py
=================
Run this in as many Colab notebooks as you want. Each one:
  - picks its own unique worker_id
  - loops: claim a job from the shared queue -> master it -> upload
    the result -> report heartbeats the whole time
  - if no job is pending, reports "idle, waiting" and polls again

There is no coordination needed between Colab instances beyond the
shared Drive folder — starting a second, third, fourth notebook with
this same script just adds another puller against the same queue.
Two workers will occasionally race for the same job (see
distributed_queue.py's docstring on the optimistic-claim limitation);
the loser just moves on to poll again, it doesn't crash or duplicate
output.

Prerequisites: run setup_colab.sh first, and fill in the service
account JSON path + Drive folder ID below (same ones used in the
Kaggle notebook).
"""

import time
import uuid

from colab_master import process_and_upscale_stream
from distributed_queue import Queue

SERVICE_ACCOUNT_JSON = "/content/service_account.json"
DRIVE_ROOT_FOLDER_ID = "PASTE_YOUR_FOLDER_ID_HERE"

POLL_INTERVAL = 5  # seconds between queue checks when idle
WORKER_ID = f"colab-worker-{uuid.uuid4().hex[:6]}"


def run_forever():
    q = Queue(SERVICE_ACCOUNT_JSON, DRIVE_ROOT_FOLDER_ID)
    print(f"{WORKER_ID} online. Polling shared queue every {POLL_INTERVAL}s.")

    while True:
        q.report_heartbeat(WORKER_ID, role="colab-mastering", stage="polling", progress=None)
        job = None
        try:
            job = q.claim_next_job(WORKER_ID)
        except Exception as e:
            print(f"[{WORKER_ID}] claim error: {e}")

        if job is None:
            time.sleep(POLL_INTERVAL)
            continue

        print(f"[{WORKER_ID}] claimed job {job['job_id']} — prompt: {job.get('prompt')!r}")
        q.report_heartbeat(
            WORKER_ID, role="colab-mastering", stage="processing_job",
            progress=0, extra={"job_id": job["job_id"]},
        )

        input_path = job.get("_local_video_path")
        output_path = f"/tmp/{job['job_id']}_master.mp4"
        try:
            process_and_upscale_stream(input_path, output_path)
            q.complete_job(job, output_path)
            q.report_heartbeat(
                WORKER_ID, role="colab-mastering", stage="job_done",
                progress=100, extra={"job_id": job["job_id"]},
            )
            print(f"[{WORKER_ID}] finished job {job['job_id']}")
        except Exception as e:
            q.report_heartbeat(
                WORKER_ID, role="colab-mastering", stage=f"job_error: {e}",
                progress=0, extra={"job_id": job["job_id"]},
            )
            print(f"[{WORKER_ID}] job {job['job_id']} failed: {e}")


if __name__ == "__main__":
    run_forever()
