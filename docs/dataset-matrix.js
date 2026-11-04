(() => {
  'use strict';
  const browser = document.querySelector('[data-dataset-browser]');
  if (!browser) return;
  const video = browser.querySelector('[data-matrix-video]');
  // The source render already encodes 0.25x physical speed; play it at rate 1.
  const playButton = browser.querySelector('[data-matrix-play]');
  const playText = playButton.querySelector('[data-play-text]');
  const playIcon = playButton.querySelector('[data-play-icon]');
  const cells = [...browser.querySelectorAll('[data-matrix-cell]')];
  const replay = browser.querySelector('[data-replay-video]');
  const replayFallback = browser.querySelector('[data-replay-fallback]');
  const title = browser.querySelector('[data-pose-title]');
  const severity = browser.querySelector('[data-pose-severity]');
  const multiplier = browser.querySelector('[data-pose-alpha]');
  const interval = browser.querySelector('[data-replay-interval]');
  const status = browser.querySelector('[data-matrix-status]');
  const names = {step: 'Step', ramp: 'Ramp', intermittent: 'Intermittent'};
  let selectionGeneration = 0;
  const formatTime = value => `${Number(value) < 0 ? '−' : '+'}${Math.abs(Number(value))}`;
  const select = (cell, focus = false) => {
    const generation = ++selectionGeneration;
    cells.forEach(item => {
      const active = item === cell;
      item.setAttribute('aria-pressed', String(active));
      item.tabIndex = active ? 0 : -1;
    });
    const alpha = Number(cell.dataset.alpha).toFixed(2);
    const description = `${names[cell.dataset.profile]} · M${cell.dataset.motor} · α = ${alpha}`;
    title.textContent = description;
    multiplier.textContent = alpha;
    severity.textContent = cell.dataset.severity;
    interval.textContent = `${formatTime(cell.dataset.startMs)} to ${formatTime(cell.dataset.endMs)} ms`;
    replay.pause();
    replay.poster = cell.dataset.preview;
    replay.src = cell.dataset.video;
    replayFallback.href = cell.dataset.video;
    replay.playbackRate = 1;
    replay.setAttribute('aria-label', `${description}: recorded-pose clip at 0.25 times physical speed. The selected motor is highlighted red.`);
    replay.load();
    replay.play().catch(error => {
      if (generation === selectionGeneration && error.name !== 'AbortError') {
        status.textContent = `${description}. Use the video controls to start the recorded replay.`;
      }
    });
    status.textContent = `${description}. ${cell.dataset.severity} fault. Ten dataset repetitions; one representative flight excerpt is replayed at 0.25 times physical speed.`;
    if (focus) cell.focus({preventScroll: true});
  };
  cells.forEach((cell, index) => {
    cell.addEventListener('click', () => select(cell));
    cell.addEventListener('keydown', event => {
      let next = index;
      if (event.key === 'ArrowRight') next = Math.min(cells.length - 1, index + 1);
      else if (event.key === 'ArrowLeft') next = Math.max(0, index - 1);
      else if (event.key === 'ArrowDown') next = Math.min(cells.length - 1, index + 12);
      else if (event.key === 'ArrowUp') next = Math.max(0, index - 12);
      else if (event.key === 'Home') next = event.ctrlKey ? 0 : Math.floor(index / 12) * 12;
      else if (event.key === 'End') next = event.ctrlKey ? cells.length - 1 : Math.floor(index / 12) * 12 + 11;
      else return;
      event.preventDefault();
      select(cells[next], true);
      cells[next].scrollIntoView({block: 'nearest', inline: 'nearest'});
    });
  });
  const updatePlayback = () => {
    playText.textContent = video.paused ? 'Play matrix' : 'Pause matrix';
    playIcon.textContent = video.paused ? '▶' : 'Ⅱ';
    playButton.setAttribute('aria-pressed', String(!video.paused));
  };
  playButton.addEventListener('click', () => {
    if (video.paused) video.play().catch(() => {
      status.textContent = 'Use Play matrix for the overview, or the replay player below for the selected flight.';
    });
    else video.pause();
  });
  video.addEventListener('play', updatePlayback);
  video.addEventListener('pause', updatePlayback);
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      video.pause();
      replay.pause();
    }
  });
  replay.playbackRate = 1;
  updatePlayback();
})();
