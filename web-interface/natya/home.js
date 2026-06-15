const hero = document.querySelector(".hero");
const canvas = document.querySelector(".hero-network");
const labels = [...document.querySelectorAll(".floating-ideas span")];
const heroContent = document.querySelector(".hero-content");

if (hero && canvas) {
  const context = canvas.getContext("2d");
  const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const seeds = [
    [0.48, 0.2, 0.09, 0.07],
    [0.29, 0.38, -0.08, 0.06],
    [0.36, 0.61, 0.07, -0.08],
    [0.69, 0.58, -0.07, -0.06],
    [0.14, 0.24, 0.05, 0.08],
    [0.84, 0.25, -0.06, 0.05],
    [0.18, 0.75, 0.07, -0.05],
    [0.82, 0.76, -0.05, -0.07],
    [0.52, 0.43, 0.06, 0.04],
    [0.58, 0.78, -0.07, 0.05],
    [0.08, 0.5, 0.06, -0.05],
    [0.92, 0.48, -0.05, 0.07],
    [0.38, 0.1, 0.08, 0.04],
    [0.66, 0.12, -0.07, 0.05],
    [0.4, 0.88, 0.05, -0.06],
    [0.7, 0.9, -0.06, -0.05],
    [0.24, 0.52, 0.05, 0.06],
    [0.76, 0.42, -0.06, -0.04],
  ];
  let width = 0;
  let height = 0;
  let lastTime = 0;
  let nodes = [];
  let exclusionZone = null;

  function resize() {
    const bounds = hero.getBoundingClientRect();
    const contentParts = [...heroContent.children].map((element) =>
      element.getBoundingClientRect(),
    );
    const scale = Math.min(devicePixelRatio, 2);
    width = bounds.width;
    height = bounds.height;
    exclusionZone = {
      left: Math.min(...contentParts.map((part) => part.left)) - bounds.left - 72,
      right:
        Math.max(...contentParts.map((part) => part.right)) - bounds.left + 72,
      top: Math.min(...contentParts.map((part) => part.top)) - bounds.top - 32,
      bottom:
        Math.max(...contentParts.map((part) => part.bottom)) - bounds.top + 32,
    };
    canvas.width = Math.round(width * scale);
    canvas.height = Math.round(height * scale);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    context.setTransform(scale, 0, 0, scale, 0, 0);

    nodes = seeds.map(([x, y, vx, vy]) => ({
      x: x * width,
      y: y * height,
      vx: vx * width * 0.01,
      vy: vy * height * 0.01,
    }));
  }

  function draw(time = 0) {
    const elapsed = Math.min((time - lastTime) / 16.67 || 1, 2);
    lastTime = time;
    context.clearRect(0, 0, width, height);

    for (const node of nodes) {
      if (!reducedMotion) {
        node.x += node.vx * elapsed;
        node.y += node.vy * elapsed;
      }

      if (node.x < 45 || node.x > width - 45) node.vx *= -1;
      if (node.y < 90 || node.y > height - 55) node.vy *= -1;
      node.x = Math.max(45, Math.min(width - 45, node.x));
      node.y = Math.max(90, Math.min(height - 55, node.y));

      if (
        node.x > exclusionZone.left &&
        node.x < exclusionZone.right &&
        node.y > exclusionZone.top &&
        node.y < exclusionZone.bottom
      ) {
        const edges = [
          [node.x - exclusionZone.left, "left"],
          [exclusionZone.right - node.x, "right"],
          [node.y - exclusionZone.top, "top"],
          [exclusionZone.bottom - node.y, "bottom"],
        ];
        const edge = edges.sort((a, b) => a[0] - b[0])[0][1];
        if (edge === "left" || edge === "right") node.vx *= -1;
        if (edge === "top" || edge === "bottom") node.vy *= -1;
        node.x =
          edge === "left"
            ? exclusionZone.left
            : edge === "right"
              ? exclusionZone.right
              : node.x;
        node.y =
          edge === "top"
            ? exclusionZone.top
            : edge === "bottom"
              ? exclusionZone.bottom
              : node.y;
      }
    }

    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const distance = Math.hypot(
          nodes[i].x - nodes[j].x,
          nodes[i].y - nodes[j].y,
        );
        if (distance > Math.min(width * 0.25, 340)) continue;

        context.strokeStyle = `rgba(232, 155, 164, ${0.32 - distance / 1400})`;
        context.lineWidth = 1;
        context.beginPath();
        context.moveTo(nodes[i].x, nodes[i].y);
        context.lineTo(nodes[j].x, nodes[j].y);
        context.stroke();
      }
    }

    nodes.slice(labels.length).forEach((node) => {
      context.fillStyle = "#efc33f";
      context.beginPath();
      context.arc(node.x, node.y, 3, 0, Math.PI * 2);
      context.fill();
    });

    labels.forEach((label, index) => {
      label.style.transform = `translate(${nodes[index].x}px, ${nodes[index].y}px)`;
    });

    if (!reducedMotion) requestAnimationFrame(draw);
  }

  new ResizeObserver(() => {
    resize();
    if (reducedMotion) draw();
  }).observe(hero);

  resize();
  draw();
}
