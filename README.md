# stock_vido service

**What it does:** takes a plain text script → builds narration audio (step1) → picks/creates stock clips (step2) → stitches and returns a **final MP4 URL** (served via Supabase Storage).

## Planned API (v1)
- `GET /health` → `{ "status": "ok", "ffmpeg": true }`
- `POST /video/orchestrate`
  - Body: `{ "script": "your text" }`
  - Response: `{ "status": "succeeded", "video_url": "https://<supabase>/storage/v1/object/public/outputs/videos/<id>.mp4" }`

## Runtime requirements
- Python 3.11.x (see `.python-version`)
- FFmpeg/FFprobe available on PATH
- Env vars set (copy `.env.example` → `.env` and fill)

## Non-goals (v1)
- No background job queue yet (single sync endpoint)
- No resumable uploads (simple upload to Supabase public bucket `outputs`)
