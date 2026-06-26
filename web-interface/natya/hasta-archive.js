import { HASTAS, mediaLabel } from "./hasta-data.js";

const archiveGrid = document.getElementById("hastaArchiveGrid");

function videoPath(hasta, fileName) {
  return encodeURI(`${hasta.mediaBasePath}/${fileName}`);
}

function renderHastaCard(hasta) {
  const selectedFile = hasta.selectedFile || hasta.mediaFiles[0];
  const article = document.createElement("article");
  article.className = "archive-card";
  article.innerHTML = `
    <div class="archive-card-media">
      <video controls muted playsinline preload="metadata">
        <source src="${videoPath(hasta, selectedFile)}" type="video/mp4" />
      </video>
      <p>Performed by ${hasta.performer}, Apsaras Dance Academy</p>
    </div>
    <div class="archive-card-copy">
      <p class="meaning-kicker">Usages of</p>
      <h2>${hasta.name}</h2>
      <img src="${hasta.image}" alt="${hasta.name} hand gesture" />
      <ul></ul>
    </div>
  `;

  const video = article.querySelector("video");
  const source = article.querySelector("source");
  const list = article.querySelector("ul");
  list.replaceChildren(...hasta.mediaFiles.map((fileName) => {
    const item = document.createElement("li");
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = hasta.labels?.[fileName] || mediaLabel(fileName);
    button.setAttribute("aria-pressed", String(fileName === selectedFile));
    button.addEventListener("click", () => {
      source.src = videoPath(hasta, fileName);
      video.load();
      video.play().catch(() => {});
      for (const option of list.querySelectorAll("button")) {
        option.setAttribute("aria-pressed", String(option === button));
      }
    });
    item.append(button);
    return item;
  }));

  return article;
}

archiveGrid.replaceChildren(...Object.values(HASTAS).map(renderHastaCard));
console.assert(archiveGrid.children.length === Object.keys(HASTAS).length);
