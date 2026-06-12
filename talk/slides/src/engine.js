// engine.js — 方向键/空格翻页, F 全屏, Home/End, hash 同步, 1280x720 等比缩放
(function () {
  const slides = Array.from(document.querySelectorAll('.slide'));
  let cur = 0;
  function fit() {
    const s = Math.min(innerWidth / 1280, innerHeight / 720);
    document.documentElement.style.setProperty('--scale', s);
  }
  function show(i, push) {
    cur = Math.max(0, Math.min(slides.length - 1, i));
    slides.forEach((el, j) => el.classList.toggle('active', j === cur));
    if (push !== false) history.replaceState(null, '', '#' + (cur + 1));
  }
  function fromHash() {
    const n = parseInt(location.hash.slice(1), 10);
    show(isNaN(n) ? 0 : n - 1, false);
  }
  addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') show(cur + 1);
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') show(cur - 1);
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(slides.length - 1);
    else if (e.key.toLowerCase() === 'f')
      document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
  });
  addEventListener('resize', fit);
  addEventListener('hashchange', fromHash);
  fit(); fromHash();
})();
