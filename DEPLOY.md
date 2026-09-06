# Deploying the SkyGuard console

You get **one link**. The ML pipeline runs *inside* the web server — there is no
separate backend to deploy. `make serve` locally and the deployed site run the
exact same `skyguard.api` app.

Measured: the container idles at ~125 MB RAM and starts in ~6 s, so any small
free tier is enough.

## Why not Vercel / Netlify

The console is a **long-running stateful process**: on startup it primes the
pipeline, then a background loop generates and scores observations forever and
pushes them to browsers over a WebSocket. Serverless functions are
request-scoped and stateless, so that loop and the in-memory history would not
survive. Use a host that runs a container.

---

## Render — free, no credit card

1. Put this project on GitHub:

   ```bash
   cd /Users/manik/Downloads/skyguard
   git remote add origin https://github.com/<you>/skyguard.git
   git push -u origin main
   ```

   (Create the empty repo at <https://github.com/new> first. It can be private
   or public.)

2. <https://render.com> → sign in with GitHub → **New → Web Service** → pick the
   repo.

3. Settings:
   - **Language:** `Docker` (Render finds the `Dockerfile`)
   - **Instance type:** `Free`
   - Leave everything else default — Render injects `$PORT` and the Dockerfile
     already reads it.

4. **Create Web Service.** First build ~4 min, then you get
   `https://skyguard-xxxx.onrender.com`. That is your link.

The free instance sleeps after 15 minutes with no traffic and takes ~50 s to
wake. See the checklist below.

## Alternative: Hugging Face Spaces

If you have an HF account: **New Space → SDK = Docker → blank → Public** is free
on the *CPU basic* hardware (the paid options are GPUs and persistent storage,
which this does not need). Then:

```bash
git remote add space https://huggingface.co/spaces/<you>/skyguard
git push space main
```

(Username + a **write** access token as the password.) The `README.md`
frontmatter configures the Space automatically.

## Alternative: Koyeb

Free, no card, deploys straight from a GitHub repo (**Create Service → GitHub →
Dockerfile → Free (Eco) instance**). Similar sleep behaviour to Render.

---

## Tuning without redeploying

- `SKYGUARD_SPEED` (env var, default `12`) — simulated 10-minute steps per real
  second. Lower it (e.g. `6`) if the charts scroll too fast for a room.

## The morning of the presentation

- [ ] Open the link **3–5 minutes early** so a sleeping instance is awake and
      the pipeline has primed. (Optional: a free UptimeRobot monitor hitting the
      URL every 10 min keeps it warm all day.)
- [ ] For a pristine baseline — full health bars, empty alert log — trigger a
      redeploy or restart from the host's dashboard and wait for it to come back.
- [ ] Load it on your **phone** and on the **venue browser** beforehand. If the
      top chip says "reconnecting", a proxy is blocking the WebSocket; the
      console then falls back to polling and still updates, just less smoothly.
- [ ] Have `docs/DEMO_SCRIPT.md` open on your phone.
- [ ] Judges can inject faults themselves from any device on the link — that is
      the point. Nothing they do persists past a restart.
