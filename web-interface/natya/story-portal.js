const page = document.body;
const apiUrl = page.dataset.storyApi;
const gate = document.getElementById("pinGate");
const pinForm = document.getElementById("pinForm");
const pinInput = document.getElementById("storyPin");
const pinError = document.getElementById("pinError");
const list = document.getElementById("storyList");
const previousStories = document.getElementById("previousStories");
const nextStories = document.getElementById("nextStories");
let stories = [];
let portalPin = sessionStorage.getItem("natya-story-portal-pin") || "";
let storyPage = 0;
const STORIES_PER_PAGE = 2;
const RASA_IMAGES = {
  Adbutham: "./assets/rasa/illustrations/adbutham.png",
  Bhayanakam: "./assets/rasa/illustrations/bhayanakam.png",
  Hasyam: "./assets/rasa/illustrations/hasyam.png",
  Karuna: "./assets/rasa/illustrations/Karuna.png",
  Raudram: "./assets/rasa/illustrations/raudram.png",
  Shantham: "./assets/rasa/illustrations/shantham.png",
  Veeram: "./assets/rasa/illustrations/Veeram.png",
};
const MOCK_STORIES = [
  { story: "I visited my thatha pati's house. I helped my pati make murukku and thattai.", rasa: "Hasyam" },
  { story: "I had a jungle themed birthday party this week. I dressed as a pink cat.", rasa: "Adbutham" },
  { story: "My brother and I had ice cream every day this week because our older sister was babysitting us.", rasa: "Shantham" },
  { story: "I learnt to throw a ball at school today. I got scared when it came towards me.", rasa: "Bhayanakam" },
  { story: "I fell down playing hopscotch and felt relaxed after resting at home.", rasa: "Karuna" },
];

function storyText(story) {
  return story.story || story.content || story.text || "No story was provided.";
}

function displayStories() {
  const pageCount = Math.ceil(stories.length / STORIES_PER_PAGE);
  const visible = stories.slice(
    storyPage * STORIES_PER_PAGE,
    (storyPage + 1) * STORIES_PER_PAGE,
  );
  list.replaceChildren();
  previousStories.hidden = nextStories.hidden = stories.length <= STORIES_PER_PAGE;
  previousStories.disabled = storyPage === 0;
  nextStories.disabled = storyPage >= pageCount - 1;
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "story-status";
    empty.textContent = "No stories have been submitted for this week yet.";
    list.append(empty);
    return;
  }
  visible.forEach((story, index) => {
    const card = document.createElement("article");
    card.className = `story-card story-card-${index % 4}`;
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    card.setAttribute("aria-pressed", "false");
    const controls = document.createElement("span");
    controls.className = "story-controls";
    controls.setAttribute("aria-hidden", "true");
    controls.textContent = "− ×";
    const copy = document.createElement("p");
    copy.textContent = storyText(story);
    const imagePath = RASA_IMAGES[story.rasa];
    if (imagePath) {
      card.classList.add("has-rasa");
      const rasa = document.createElement("div");
      rasa.className = "story-rasa";
      const rasaImage = document.createElement("img");
      rasaImage.src = imagePath;
      rasaImage.alt = `${story.rasa} rasa`;
      const rasaName = document.createElement("span");
      rasaName.textContent = story.rasa;
      rasa.append(rasaImage, rasaName);
      card.append(controls, rasa, copy);
    } else {
      card.append(controls, copy);
    }
    const toggle = () => {
      const selected = card.classList.toggle("is-selected");
      card.setAttribute("aria-pressed", String(selected));
    };
    card.addEventListener("click", toggle);
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggle();
      }
    });
    list.append(card);
  });
}

async function loadStories() {
  try {
    const response = await fetch(apiUrl, {
      headers: { Accept: "application/json", "X-Story-Pin": portalPin },
    });
    if (!response.ok) throw new Error("Stories could not be loaded.");
    const payload = await response.json();
    stories = Array.isArray(payload) ? payload : payload.stories || [];
    displayStories();
  } catch (error) {
    // ponytail: mock stories keep the portal reviewable until the API is deployed.
    stories = MOCK_STORIES;
    displayStories();
  }
}

// ponytail: this only gates the static teacher view; enforce the PIN on the API before sharing real stories.
function unlock(event) {
  event.preventDefault();
  if (pinInput.value !== "0000") {
    pinError.textContent = "That PIN is not correct.";
    pinInput.select();
    return;
  }
  portalPin = pinInput.value;
  sessionStorage.setItem("natya-story-portal-unlocked", "true");
  sessionStorage.setItem("natya-story-portal-pin", portalPin);
  gate.hidden = true;
  loadStories();
}

console.assert(storyText({ content: "A story" }) === "A story");

previousStories.addEventListener("click", () => {
  storyPage -= 1;
  displayStories();
});
nextStories.addEventListener("click", () => {
  storyPage += 1;
  displayStories();
});

pinForm.addEventListener("submit", unlock);
if (sessionStorage.getItem("natya-story-portal-unlocked") === "true" && portalPin) {
  gate.hidden = true;
  loadStories();
}
