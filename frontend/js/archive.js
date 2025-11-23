async function fetchArchive() {
  const listEl = document.getElementById('archive-list');
  listEl.textContent = 'Loading…';
  try {
    const resp = await fetch('/api/archive');
    if (!resp.ok) throw new Error('Failed to load archive');
    const items = await resp.json();
    listEl.textContent = '';
    if (!items.length) {
      listEl.textContent = 'No farts yet. Be the first!';
      return;
    }
    items.forEach((item) => {
      const container = document.createElement('div');
      container.className = 'archive-item card';

      const audio = document.createElement('audio');
      audio.controls = true;
      audio.src = item.url;

      const meta = document.createElement('div');
      const date = new Date(item.created_at);
      meta.textContent = `Recorded at ${date.toLocaleString()}`;

      container.appendChild(audio);
      container.appendChild(meta);
      listEl.appendChild(container);
    });
  } catch (err) {
    console.error(err);
    listEl.textContent = 'Could not load archive.';
  }
}

document.addEventListener('DOMContentLoaded', fetchArchive);
