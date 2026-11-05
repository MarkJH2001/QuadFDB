(() => {
  'use strict';
  const browser = document.querySelector('[data-dataset-browser]');
  if (!browser) return;
  const video = browser.querySelector('[data-matrix-video]');
  // The source render already encodes 0.25x physical speed; play it at rate 1.
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
  video.muted = true;
  video.defaultMuted = true;
  video.autoplay = true;
  video.playsInline = true;
  video.loop = true;
  video.preload = 'metadata';
  video.playbackRate = 1;
  let matrixVisible = false;
  let replayWasPlaying = false;
  const updatePlayback = () => {
    if (document.hidden || !matrixVisible) {
      video.pause();
      return;
    }
    video.play().catch(() => {
      // Muted autoplay can still be deferred by browser resource policies.
      // Readiness, visibility and the next matrix interaction retry it.
    });
  };
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      matrixVisible = entries.some(entry => entry.isIntersecting);
      updatePlayback();
    });
    observer.observe(video);
  } else {
    const updateVisibility = () => {
      const bounds = video.getBoundingClientRect();
      matrixVisible = bounds.bottom > 0 && bounds.top < window.innerHeight;
      updatePlayback();
    };
    window.addEventListener('scroll', updateVisibility, {passive: true});
    window.addEventListener('resize', updateVisibility, {passive: true});
    updateVisibility();
  }
  video.addEventListener('canplay', updatePlayback);
  video.addEventListener('playing', () => {
    if (document.hidden || !matrixVisible) video.pause();
  });
  browser.addEventListener('pointerdown', updatePlayback, {passive: true});
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      replayWasPlaying = !replay.paused;
      video.pause();
      replay.pause();
    } else {
      updatePlayback();
      if (replayWasPlaying) replay.play().catch(() => {});
    }
  });
  replay.playbackRate = 1;
})();
