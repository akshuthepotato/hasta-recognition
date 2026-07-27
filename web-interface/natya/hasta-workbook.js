import { HASTAS as HASTA_DATA } from "./hasta-data.js";

const HASTAS = HASTA_DATA;
const WORDS = [
  "Sun", "Moon", "Star", "Rain", "Wind", "River", "Ocean", "Mountain",
  "Tree", "Flower", "Leaf", "Seed", "Fire", "Cloud", "Rock", "Sky",
  "Forest", "Water", "Bird", "Fish", "Elephant", "Peacock", "Butterfly",
  "Snake", "Deer", "Lion", "Cat", "Dog", "Rabbit", "Monkey", "Turtle",
  "Horse", "Bee", "Mother", "Father", "Friend", "Teacher", "Baby",
  "Dancer", "King", "Queen", "Child", "Family", "Run", "March", "Jump",
  "Sleep", "Leap", "Dance", "Skip", "Swing", "Eat", "Drink", "Read",
  "Write", "Fly", "Swim", "Wave", "Hug", "Book", "Crown", "Lamp",
  "Mirror", "Flower Pot", "Ring", "Umbrella", "Door", "Window", "Boat",
  "Home", "School", "Temple", "Park", "Beach", "Garden", "Stage",
  "Playground", "Jungle", "Village",
];

const repositories = document.getElementById("hastaRepositories");
const wordBank = document.getElementById("wordBank");
const matchStatus = document.getElementById("matchStatus");
const diaryHastaImage = document.getElementById("diaryHastaImage");
const diaryHastaName = document.getElementById("diaryHastaName");
const diaryPageLabel = document.getElementById("diaryPageLabel");
const interpretationList = document.getElementById("interpretationList");
const interpretationDialog = document.getElementById("interpretationDialog");
const interpretationForm = document.getElementById("interpretationForm");
const interpretationText = document.getElementById("interpretationText");
const diaryHastas = Object.entries(HASTAS);
const storageKey = "natya-hasta-diary-v1";
let selectedWord = null;
let diaryPage = 0;
let diaryEntries = loadDiaryEntries();

function loadDiaryEntries() {
  try {
    return JSON.parse(localStorage.getItem(storageKey)) || {};
  } catch {
    return {};
  }
}

function saveDiaryEntries() {
  localStorage.setItem(storageKey, JSON.stringify(diaryEntries));
}

function words() {
  return [...document.querySelectorAll(".word-chip")];
}

function updateCount() {
  const matched = words().filter((word) => word.closest(".hasta-repository")).length;
  if (matched === WORDS.length) matchStatus.textContent = "All words placed. Review your interpretation or reset to begin again.";
}

function selectWord(word) {
  selectedWord = word;
  for (const chip of words()) chip.setAttribute("aria-pressed", String(chip === word));
  matchStatus.textContent = `Selected ${word.textContent}. Choose a hasta repository.`;
}

function placeWord(word, repository) {
  if (!word) return;
  repository.querySelector(".repository-words").append(word);
  selectedWord = null;
  word.setAttribute("aria-pressed", "false");
  matchStatus.textContent = `${word.textContent} placed with ${repository.dataset.hasta}.`;
  updateCount();
}

function render() {
  repositories.replaceChildren(...Object.values(HASTAS).map((hasta) => {
    const card = document.createElement("article");
    card.className = "hasta-repository";
    card.dataset.hasta = hasta.name;
    card.tabIndex = 0;
    card.innerHTML = `<img src="${hasta.image}" alt="${hasta.name} hand gesture" /><h2>${hasta.name}</h2><div class="repository-words" aria-label="Words placed with ${hasta.name}"></div>`;
    card.addEventListener("dragover", (event) => { event.preventDefault(); card.classList.add("is-drop-target"); });
    card.addEventListener("dragleave", () => card.classList.remove("is-drop-target"));
    card.addEventListener("drop", (event) => {
      event.preventDefault();
      card.classList.remove("is-drop-target");
      placeWord(document.getElementById(event.dataTransfer.getData("text/plain")), card);
    });
    card.addEventListener("click", () => placeWord(selectedWord, card));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); placeWord(selectedWord, card); }
    });
    return card;
  }));

  wordBank.replaceChildren(...WORDS.map((word, index) => {
    const chip = document.createElement("button");
    chip.id = `word-${index}`;
    chip.className = "word-chip";
    chip.type = "button";
    chip.draggable = true;
    chip.textContent = word;
    chip.setAttribute("aria-pressed", "false");
    chip.addEventListener("click", () => selectWord(chip));
    chip.addEventListener("dragstart", (event) => event.dataTransfer.setData("text/plain", chip.id));
    return chip;
  }));
  updateCount();
}

function renderDiary() {
  const [key, hasta] = diaryHastas[diaryPage];
  const entries = diaryEntries[key] || [];
  diaryHastaImage.src = hasta.image;
  diaryHastaImage.alt = `${hasta.name} hand gesture`;
  diaryHastaName.textContent = hasta.name;
  diaryPageLabel.textContent = `${diaryPage + 1} / ${diaryHastas.length}`;
  document.getElementById("previousDiaryPage").disabled = diaryPage === 0;
  document.getElementById("nextDiaryPage").disabled = diaryPage === diaryHastas.length - 1;
  interpretationList.replaceChildren(...entries.map((entry) => {
    const item = document.createElement("li");
    item.textContent = entry;
    return item;
  }));
}

document.getElementById("resetMatch").addEventListener("click", render);
document.getElementById("previousDiaryPage").addEventListener("click", () => {
  diaryPage -= 1;
  renderDiary();
});
document.getElementById("nextDiaryPage").addEventListener("click", () => {
  diaryPage += 1;
  renderDiary();
});
document.getElementById("addInterpretation").addEventListener("click", () => {
  interpretationText.value = "";
  interpretationDialog.showModal();
  interpretationText.focus();
});
document.querySelector(".dialog-close").addEventListener("click", () => interpretationDialog.close());
interpretationForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const entry = interpretationText.value.trim();
  if (!entry) return;
  const [key] = diaryHastas[diaryPage];
  diaryEntries[key] = [...(diaryEntries[key] || []), entry];
  saveDiaryEntries();
  interpretationDialog.close();
  renderDiary();
});

render();
renderDiary();
console.assert(Object.keys(HASTAS).length === 24 && WORDS.length === 80);
