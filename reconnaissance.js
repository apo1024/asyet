/* Asyèt — reconnaissance des plats sur l'appareil (CLIP ViT-B/32, OpenAI, licence MIT ; ONNX 4 bits), sans compte ni envoi de photo.
   La photo est comparée à la description de chaque plat de plats.json (signatures précalculées dans plats-emb.json). */
const Asyet = (() => {
  let sess = null, PLATS = null, EMB = null;
  async function load() {
    if (!PLATS) {
      PLATS = await (await fetch('plats.json')).json();
      const e = await (await fetch('plats-emb.json')).json();
      EMB = e.emb.map(v => Float32Array.from(v));
    }
    if (sess) return sess;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
    ort.env.wasm.proxy = true;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
    sess = await ort.InferenceSession.create('clip-vision-q4.onnx', { executionProviders: ['wasm'] });
    return sess;
  }
  const M = [0.48145466, 0.4578275, 0.40821073], S = [0.26862954, 0.26130258, 0.27577711];
  function tensor(img, sx, sy, sw, sh) {
    const k = 224 / Math.min(sw, sh), rw = sw * k, rh = sh * k;
    const c = document.createElement('canvas'); c.width = 224; c.height = 224;
    const g = c.getContext('2d'); g.imageSmoothingQuality = 'high';
    g.drawImage(img, sx, sy, sw, sh, -(rw - 224) / 2, -(rh - 224) / 2, rw, rh);
    const px = g.getImageData(0, 0, 224, 224).data, n = 224 * 224, x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) for (let ch = 0; ch < 3; ch++) x[ch * n + i] = (px[i * 4 + ch] / 255 - M[ch]) / S[ch];
    return new ort.Tensor('float32', x, [1, 3, 224, 224]);
  }
  async function scores(img, sx, sy, sw, sh) {
    const out = await sess.run({ pixel_values: tensor(img, sx, sy, sw, sh) });
    const v = out.image_embeds.data; let nrm = 0; for (let i = 0; i < v.length; i++) nrm += v[i] * v[i]; nrm = Math.sqrt(nrm) || 1;
    return EMB.map(e => { let s = 0; for (let i = 0; i < e.length; i++) s += e[i] * v[i]; return s / nrm; });
  }
  /* image entière + 4 quarts : une assiette contient souvent plusieurs plats */
  async function suggest(img, n = 6) {
    await load();
    const W = img.naturalWidth || img.width, H = img.naturalHeight || img.height;
    const zones = [[0, 0, W, H], [0, 0, W * .6, H * .6], [W * .4, 0, W * .6, H * .6], [0, H * .4, W * .6, H * .6], [W * .4, H * .4, W * .6, H * .6]];
    /* les 2 meilleurs plats de chaque zone d'abord (diversité), puis les meilleurs au total */
    const best = new Array(PLATS.length).fill(-1), pick = [];
    for (const [x, y, w, h] of zones) {
      const s = await scores(img, x, y, w, h);
      s.forEach((v, i) => { if (v > best[i]) best[i] = v; });
      s.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 2).forEach(([, i]) => { if (!pick.includes(i)) pick.push(i); });
    }
    const rest = best.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).map(([, i]) => i).filter(i => !pick.includes(i));
    return pick.sort((a, b) => best[b] - best[a]).concat(rest).slice(0, n).map(i => PLATS[i]);
  }
  function search(q) {
    const t = q.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
    const f = s => s.toLowerCase().normalize('NFD').replace(/[̀-ͯ]/g, '');
    return (PLATS || []).filter(p => f(p.fr).includes(t) || f(p.ht).includes(t)).slice(0, 8);
  }
  const item = p => { const fr = p.fr.replace(/\s*\(.*\)$/, ''); return { name: fr.toLowerCase() === p.ht.toLowerCase() ? p.ht : p.ht + ' — ' + fr, grams: p.g, kcal100: p.kcal, p100: p.p, c100: p.c, f100: p.f, thumb: null }; };
  return { load, suggest, search, item, plats: () => PLATS };
})();
