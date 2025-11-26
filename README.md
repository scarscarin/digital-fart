# Digital Fart

## Pushing this project to GitHub
1. Create a new repository on GitHub (do **not** initialize it with a README or .gitignore to avoid conflicts).
2. From this folder, point your existing repo at GitHub:
   ```bash
   git remote add origin git@github.com:<your-username>/<your-repo>.git
   git branch -M main
   git push -u origin main
   ```
   If you prefer HTTPS, swap the remote URL to `https://github.com/<your-username>/<your-repo>.git`.

## Working with binary assets
If you see a "Binary files are not supported" error while trying to upload through the GitHub web UI, push the files with Git instead of the browser. For large or frequently changing binaries (e.g., videos, audio samples), set up Git LFS first:

```bash
# Install Git LFS (one time per machine)
git lfs install

# Track the binary types you need
cd /workspace/digital-fart
git lfs track "*.webm"
git lfs track "*.mp4"
git lfs track "*.wav"

# Commit the .gitattributes file Git LFS creates
git add .gitattributes
```

After tracking, add and commit your binaries normally. When you push, Git LFS will handle the large files automatically:

```bash
git add <your files>
git commit -m "Add binary assets"
git push
```

## Quick status & test commands
- Check what will be pushed: `git status -sb`
- Run the backend syntax check used in CI: `python -m compileall backend`
