const start = document.querySelector('#start');
const detail = document.querySelector('#load-detail');
const status = document.querySelector('#status');
const seed = document.querySelector('#seed');
const incoming = new URL(location.href).searchParams.get('seed');
if (incoming && /^\d+$/.test(incoming) && Number(incoming) <= 4294967295) seed.value = incoming;
document.querySelector('#seed-form').addEventListener('submit', event => {
  event.preventDefault();
  if (!document.querySelector('#seed-form').reportValidity()) return;
  const next = new URL(location.href);
  next.searchParams.set('seed', seed.value);
  location.href = next.href;
});
start.addEventListener('click', async () => {
  start.disabled = true;
  start.textContent = 'Loading…';
  detail.textContent = 'Checking WebGPU…';
  try {
    if (!navigator.gpu) throw new Error('WebGPU is unavailable. Try a current desktop Chrome, Edge or Safari with hardware acceleration enabled.');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('No WebGPU adapter is available. Enable hardware acceleration or try another computer.');
    detail.textContent = 'Downloading the app…';
    const { default: init } = await import('./pkg/map_generation_lab.js');
    const response = await fetch('./pkg/map_generation_lab_bg.wasm');
    if (!response.ok) throw new Error(`App download failed (${response.status}). Please reload.`);
    detail.textContent = 'Loading the app and growing the world…';
    status.textContent = 'Generating the world…';
    // Let wasm-bindgen stream the response into WebAssembly instead of keeping
    // a second copy of the complete binary in JavaScript memory.
    await init({ module_or_path: response });
  } catch (error) {
    console.error(error);
    detail.textContent = error.message || String(error);
    status.textContent = 'The demo could not start.';
    start.disabled = false;
    start.textContent = 'Try again';
  }
});
window.addEventListener('error', event => {
  detail.textContent = event.message || 'The app encountered an error. Please reload.';
  status.textContent = 'The demo encountered an error.';
});
