"""
distributed_queue.py
=====================
Shared coordination layer for the Kaggle (generation) + N-Colab
(mastering) pipeline. Paste this SAME file into the Kaggle notebook and
into every Colab worker notebook — it's the thing that lets them all
see each other as one system instead of running in isolation.

WHY GOOGLE DRIVE + A SERVICE ACCOUNT:
Kaggle and Colab can't reach each other directly (no shared network).
Google Drive is a coordination point both can reach headlessly. Using a
service account (rather than the interactive `google.colab.auth`/
`drive.mount()` flows) means BOTH platforms authenticate identically —
Kaggle has no interactive OAuth browser popup, so anything that only
works via `drive.mount()` breaks there.

ONE-TIME SETUP (do this once, outside any notebook):
  1. Google Cloud Console -> IAM & Admin -> Service Accounts -> Create.
  2. Create a JSON key for it, download it.
  3. Create a folder in your own Google Drive (e.g. "video_pipeline"),
     right-click -> Share -> paste the service account's email
     (looks like ...@...iam.gserviceaccount.com) -> give it Editor.
  4. Copy that folder's ID from its Drive URL
     (https://drive.google.com/drive/folders/<THIS_PART>).
  5. Upload the JSON key file into each Kaggle/Colab session as a
     Secret/private file — never commit it to a public notebook.

Both notebooks then do:
    from distributed_queue import Queue
    q = Queue(service_account_json="/path/to/key.json", root_folder_id="...")

HONEST LIMITATION:
Job claiming here uses an optimistic rename-then-verify check, not a
real transactional lock. It's solid for a handful of cooperating
workers (the failure window is a couple of seconds around claim time,
and losing a race just means retrying the next poll) — it is NOT
safe against adversarial or very high-concurrency use. If you need
that, put a proper database (Firestore, etc.) behind this instead of
raw Drive files.
"""

import io
import json
import time
import uuid

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from google.oauth2 import service_account

SCOPES = ["https://www.googleapis.com/auth/drive"]
HEARTBEAT_STALE_AFTER = 20  # seconds


class Queue:
    def __init__(self, service_account_json, root_folder_id):
        creds = service_account.Credentials.from_service_account_file(
            service_account_json, scopes=SCOPES
        )
        self.svc = build("drive", "v3", credentials=creds)
        self.root_id = root_folder_id
        self._folder_ids = {}
        for name in ("pending", "claimed", "done", "status"):
            self._folder_ids[name] = self._ensure_subfolder(name)

    # ---------- low-level Drive helpers ----------
    def _ensure_subfolder(self, name):
        q = (
            f"'{self.root_id}' in parents and name='{name}' "
            "and mimeType='application/vnd.google-apps.folder' and trashed=false"
        )
        res = self.svc.files().list(q=q, fields="files(id)").execute()
        files = res.get("files", [])
        if files:
            return files[0]["id"]
        meta = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
            "parents": [self.root_id],
        }
        return self.svc.files().create(body=meta, fields="id").execute()["id"]

    def _upload_json(self, folder_id, filename, data):
        media = MediaIoBaseUpload(
            io.BytesIO(json.dumps(data).encode()), mimetype="application/json"
        )
        existing = self._find(folder_id, filename)
        if existing:
            return self.svc.files().update(fileId=existing["id"], media_body=media).execute()
        meta = {"name": filename, "parents": [folder_id]}
        return self.svc.files().create(body=meta, media_body=media, fields="id").execute()

    def _upload_file(self, folder_id, filename, local_path):
        media = MediaIoBaseUpload(open(local_path, "rb"), mimetype="application/octet-stream")
        meta = {"name": filename, "parents": [folder_id]}
        return self.svc.files().create(body=meta, media_body=media, fields="id").execute()

    def _download_file(self, file_id, local_path):
        request = self.svc.files().get_media(fileId=file_id)
        with io.FileIO(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

    def _read_json(self, file_id):
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, self.svc.files().get_media(fileId=file_id))
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return json.loads(buf.getvalue().decode())

    def _find(self, folder_id, filename):
        q = f"'{folder_id}' in parents and name='{filename}' and trashed=false"
        res = self.svc.files().list(q=q, fields="files(id,name)").execute()
        files = res.get("files", [])
        return files[0] if files else None

    def _list(self, folder_id):
        q = f"'{folder_id}' in parents and trashed=false"
        res = self.svc.files().list(
            q=q, fields="files(id,name,createdTime)", orderBy="createdTime"
        ).execute()
        return res.get("files", [])

    def _move(self, file_id, new_parent_id, old_parent_id):
        self.svc.files().update(
            fileId=file_id, addParents=new_parent_id, removeParents=old_parent_id, fields="id"
        ).execute()

    def _rename(self, file_id, new_name):
        self.svc.files().update(fileId=file_id, body={"name": new_name}).execute()

    # ---------- job queue ----------
    def submit_job(self, prompt, video_local_path=None, extra=None):
        """Kaggle side: push a finished raw clip + its metadata into
        pending/. Returns the job_id."""
        job_id = str(uuid.uuid4())[:8]
        video_file_id = None
        if video_local_path:
            uploaded = self._upload_file(
                self._folder_ids["pending"], f"{job_id}_video.mp4", video_local_path
            )
            video_file_id = uploaded["id"]

        job = {
            "job_id": job_id,
            "prompt": prompt,
            "video_file_id": video_file_id,
            "created_at": time.time(),
            "extra": extra or {},
        }
        self._upload_json(self._folder_ids["pending"], f"{job_id}.json", job)
        return job_id

    def claim_next_job(self, worker_id):
        """Colab side: try to take the oldest pending job. Optimistic
        claim: rename the job file to include our worker_id, then
        re-check it still shows our worker_id (nobody else raced us)
        before treating it as ours. Returns a job dict or None."""
        pending = [f for f in self._list(self._folder_ids["pending"]) if f["name"].endswith(".json")]
        if not pending:
            return None

        candidate = pending[0]  # oldest by createdTime
        claim_name = f"claimed_by_{worker_id}__{candidate['name']}"
        self._rename(candidate["id"], claim_name)
        time.sleep(1.5)  # let any concurrent renamer's write land

        current = self.svc.files().get(fileId=candidate["id"], fields="name").execute()
        if current["name"] != claim_name:
            # Someone else renamed it after us — we lost the race.
            return None

        self._move(candidate["id"], self._folder_ids["claimed"], self._folder_ids["pending"])
        job = self._read_json(candidate["id"])
        job["_json_file_id"] = candidate["id"]

        if job.get("video_file_id"):
            local_path = f"/tmp/{job['job_id']}_input.mp4"
            self._download_file(job["video_file_id"], local_path)
            job["_local_video_path"] = local_path

        return job

    def complete_job(self, job, output_local_path):
        """Colab side: upload the finished master and retire the job
        record."""
        self._upload_file(
            self._folder_ids["done"], f"{job['job_id']}_master.mp4", output_local_path
        )
        self._move(job["_json_file_id"], self._folder_ids["done"], self._folder_ids["claimed"])

    # ---------- heartbeat / status ----------
    def report_heartbeat(self, node_id, role, stage, progress=None, extra=None):
        """Every node (kaggle-gpu0, kaggle-gpu1, colab-worker-<uuid>...)
        calls this every couple of seconds. Same call shape regardless
        of platform, so the merged status view treats all nodes
        uniformly."""
        payload = {
            "node_id": node_id,
            "role": role,          # "kaggle-generator" | "colab-mastering"
            "stage": stage,
            "progress": progress,
            "extra": extra or {},
            "last_heartbeat": time.time(),
        }
        self._upload_json(self._folder_ids["status"], f"{node_id}.json", payload)

    def list_all_status(self):
        """Reads every node's latest heartbeat. This is what makes the
        whole thing feel like one system: a single call shows every
        Kaggle GPU and every currently-running Colab worker, however
        many of the latter you've started, with staleness flagged."""
        now = time.time()
        result = {}
        for f in self._list(self._folder_ids["status"]):
            data = self._read_json(f["id"])
            age = now - data["last_heartbeat"]
            data["stale"] = age > HEARTBEAT_STALE_AFTER
            data["age_seconds"] = round(age, 1)
            result[data["node_id"]] = data
        return result
