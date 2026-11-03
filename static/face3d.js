/* 3D face map: the stitched face texture on the shared face mesh, with the unique spots as dots. Needs three.js (vendor/three.min.js).
   FaceVeda3D.mount(el, data, atlasUrl)  data = {vertices:[x,y,z...] (face width 1, y down, z away from the camera), tris:[a,b,c...], uv:[u,v...], spots:[{p:[x,y,z], zone, n}]}.
   Drag to turn, wheel/pinch to zoom, double-click to reset. Dots: green = seen in 2+ photos, amber = seen in one. */
window.FaceVeda3D = (function () {
  function mount(el, data, atlasUrl, opts) {
    opts = opts || {}; const W = el.clientWidth || 480, H = opts.height || 600; el.style.position = "relative"; el.style.touchAction = "none";
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); renderer.setPixelRatio(Math.min(3, Math.max(2, window.devicePixelRatio || 1))); renderer.setSize(W, H); el.appendChild(renderer.domElement);
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(30, W / H, 0.1, 20); cam.position.set(0, 0, 2.9);
    const V = data.vertices, n = V.length / 3, pos = new Float32Array(n * 3), uv = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) { pos[3*i] = V[3*i]; pos[3*i+1] = -V[3*i+1]; pos[3*i+2] = -V[3*i+2]; uv[2*i] = data.uv[2*i]; uv[2*i+1] = 1 - data.uv[2*i+1]; }   // photo axes (y down, z away) -> three axes (y up, z to the viewer)
    const g = new THREE.BufferGeometry(); g.setAttribute("position", new THREE.BufferAttribute(pos, 3)); g.setAttribute("uv", new THREE.BufferAttribute(uv, 2)); g.setIndex(data.tris); g.computeVertexNormals();
    const tex = new THREE.TextureLoader().load(atlasUrl, () => render()); tex.anisotropy = renderer.capabilities.getMaxAnisotropy(); tex.minFilter = THREE.LinearMipmapLinearFilter; tex.magFilter = THREE.LinearFilter;
    const face = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide })); const root = new THREE.Group(); root.add(face); scene.add(root);
    const cy = -(Math.min.apply(null, V.filter((_, i) => i % 3 == 1)) + Math.max.apply(null, V.filter((_, i) => i % 3 == 1))) / 2; face.position.y = -cy; face.position.z = 0.1;
    const dots = new THREE.Group(); face.add(dots); const sg = new THREE.SphereGeometry(0.009, 12, 12);
    (data.spots || []).forEach(s => { if (!s.p) return; const m = new THREE.Mesh(sg, new THREE.MeshBasicMaterial({ color: s.n >= 2 ? 0x1fa35c : 0xf0a020 })); m.position.set(s.p[0], -s.p[1], -s.p[2] + 0.012); dots.add(m); });
    let yaw = 0, pitch = 0, zoom = 1, drag = null; const set = () => { root.rotation.y = yaw; root.rotation.x = pitch; cam.position.z = 2.9 / zoom; render(); };
    function render() { renderer.render(scene, cam); }
    const c = renderer.domElement; c.style.cursor = "grab"; c.style.width = "100%"; c.style.height = H + "px";
    c.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, yaw, pitch]; c.setPointerCapture(e.pointerId); c.style.cursor = "grabbing"; });
    c.addEventListener("pointermove", e => { if (!drag) return; yaw = drag[2] + (e.clientX - drag[0]) * 0.008; pitch = Math.max(-0.9, Math.min(0.9, drag[3] + (e.clientY - drag[1]) * 0.008)); set(); });
    c.addEventListener("pointerup", () => { drag = null; c.style.cursor = "grab"; });
    c.addEventListener("wheel", e => { e.preventDefault(); zoom = Math.max(0.6, Math.min(6, zoom * (e.deltaY < 0 ? 1.12 : 0.89))); set(); }, { passive: false });
    c.addEventListener("dblclick", () => { yaw = 0; pitch = 0; zoom = 1; set(); }); set();
    return { setSpotsVisible: v => { dots.visible = v; render(); }, render, dispose: () => { renderer.dispose(); c.remove(); } };
  }

  /* Flat view: the stitched texture at full resolution, spots marked, drag to pan, wheel to zoom, double-click to reset. Best place to check an individual spot. */
  function mountFlat(el, data, atlasUrl, opts) {
    opts = opts || {}; const W = el.clientWidth || 480, H = opts.height || 600, dpr = Math.min(2, window.devicePixelRatio || 1);
    const c = document.createElement("canvas"); c.width = W * dpr; c.height = H * dpr; c.style.width = "100%"; c.style.height = H + "px"; c.style.cursor = "grab"; c.style.touchAction = "none"; c.style.background = "#222"; el.appendChild(c);
    const g = c.getContext("2d"), img = new Image(); let k = 1, ox = 0, oy = 0, show = true, drag = null, fit = 1;
    function draw() {
      g.setTransform(1, 0, 0, 1, 0, 0); g.clearRect(0, 0, c.width, c.height); if (!img.width) return; g.imageSmoothingQuality = "high";
      g.setTransform(dpr * k * fit, 0, 0, dpr * k * fit, dpr * ox, dpr * oy); g.drawImage(img, 0, 0);
      if (show) (data.spots || []).forEach(s => { if (!s.uv) return; const x = s.uv[0] * img.width, y = s.uv[1] * img.height; g.beginPath(); g.arc(x, y, Math.max(7, img.width / 260), 0, 6.283);
        g.lineWidth = Math.max(2, img.width / 900); g.strokeStyle = s.n >= 2 ? "#1fa35c" : "#f0a020"; g.stroke(); });
    }
    img.onload = () => { fit = Math.min(W / img.width, H / img.height); ox = (W - img.width * fit) / 2; oy = (H - img.height * fit) / 2; draw(); }; img.src = atlasUrl;
    const reset = () => { k = 1; ox = (W - img.width * fit) / 2; oy = (H - img.height * fit) / 2; draw(); };
    c.addEventListener("pointerdown", e => { drag = [e.clientX, e.clientY, ox, oy]; c.setPointerCapture(e.pointerId); c.style.cursor = "grabbing"; });
    c.addEventListener("pointermove", e => { if (!drag) return; ox = drag[2] + e.clientX - drag[0]; oy = drag[3] + e.clientY - drag[1]; draw(); });
    c.addEventListener("pointerup", () => { drag = null; c.style.cursor = "grab"; });
    c.addEventListener("wheel", e => { e.preventDefault(); const r = c.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, f = e.deltaY < 0 ? 1.2 : 1 / 1.2, nk = Math.max(1, Math.min(12, k * f)), q = nk / k;
      ox = px - (px - ox) * q; oy = py - (py - oy) * q; k = nk; draw(); }, { passive: false });
    c.addEventListener("dblclick", reset);
    return { setSpotsVisible: v => { show = v; draw(); }, draw };
  }
  return { mount, mountFlat };
})();
