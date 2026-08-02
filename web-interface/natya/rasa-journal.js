const journal = document.querySelector(".rasa-journal");
const journalStatus = document.getElementById("journalStatus");
const storyInput = document.getElementById("story");
const submitStory = document.getElementById("submitStory");
const selectedRasaPanel = document.getElementById("selectedRasa");
const selectedRasaImage = document.getElementById("selectedRasaImage");
const selectedRasaName = document.getElementById("selectedRasaName");
let selectedRasa = "";

function selectRasa(rasa) {
  selectedRasa = rasa;
  const option = document.querySelector(`.rasa-option[data-rasa="${rasa}"]`);
  const image = option.querySelector("img");
  selectedRasaImage.src = image.src;
  selectedRasaImage.alt = image.alt;
  selectedRasaName.textContent = option.querySelector("span").textContent;
  selectedRasaPanel.hidden = false;
  document.querySelectorAll(".rasa-option").forEach((option) => {
    option.classList.toggle("is-selected", option.dataset.rasa === rasa);
  });
  journal.querySelector(".rasa-story-entry").classList.add("has-rasa");
  journalStatus.textContent = `${rasa} selected.`;
}

document.querySelectorAll(".rasa-option").forEach((option) => {
  option.addEventListener("click", () => selectRasa(option.dataset.rasa));
  option.addEventListener("dragstart", (event) => {
    event.dataTransfer.setData("text/plain", option.dataset.rasa);
  });
});

storyInput.addEventListener("dragover", (event) => {
  event.preventDefault();
  storyInput.classList.add("is-drop-target");
});
storyInput.addEventListener("dragleave", () => storyInput.classList.remove("is-drop-target"));
storyInput.addEventListener("drop", (event) => {
  event.preventDefault();
  storyInput.classList.remove("is-drop-target");
  selectRasa(event.dataTransfer.getData("text/plain"));
});

submitStory.addEventListener("click", async () => {
  const story = storyInput.value.trim();
  if (!story) {
    journalStatus.textContent = "Write your story before submitting.";
    storyInput.focus();
    return;
  }
  submitStory.disabled = true;
  try {
    const response = await fetch(journal.dataset.storyApi, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ story, rasa: selectedRasa || null }),
    });
    if (!response.ok) throw new Error();
    storyInput.value = "";
    selectedRasa = "";
    selectedRasaPanel.hidden = true;
    journal.querySelector(".rasa-story-entry").classList.remove("has-rasa");
    document.querySelectorAll(".rasa-option").forEach((option) => option.classList.remove("is-selected"));
    journalStatus.textContent = "Your story has been saved.";
  } catch {
    journalStatus.textContent = "Your story could not be saved. Please try again.";
  } finally {
    submitStory.disabled = false;
  }
});

console.assert(storyInput.id === "story");
