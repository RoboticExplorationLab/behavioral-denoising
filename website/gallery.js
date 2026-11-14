"use strict";

(() => {
  const grid = document.querySelector("#trial-grid");
  const resultCount = document.querySelector("#result-count");
  const search = document.querySelector("#seed-search");
  const emptyState = document.querySelector("#empty-state");
  const loadError = document.querySelector("#load-error");
  let cards = [];

  const percentage = (value) => `${(value * 100).toFixed(1)}%`;
  const element = (tag, className, content) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (content !== undefined) node.textContent = content;
    return node;
  };

  function pauseOtherVideos(activeVideo) {
    for (const video of grid.querySelectorAll("video")) {
      if (video !== activeVideo) video.pause();
    }
  }

  function createCard(trial, index) {
    const card = element("article", "trial-card");
    card.setAttribute("aria-labelledby", `seed-${trial.seed}`);
    const media = element("div", "trial-media");
    const poster = element("img", "trial-poster");
    poster.src = trial.poster;
    poster.alt = `Push-T trajectory preview for seed ${trial.seed}`;
    poster.loading = index < 3 ? "eager" : "lazy";
    poster.decoding = "async";
    const playButton = element("button", "play-button");
    playButton.type = "button";
    playButton.setAttribute("aria-label", `Play trial, seed ${trial.seed}, peak coverage ${percentage(trial.peak_coverage)}`);
    const playIcon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    playIcon.setAttribute("viewBox", "0 0 12 14");
    playIcon.setAttribute("aria-hidden", "true");
    const triangle = document.createElementNS("http://www.w3.org/2000/svg", "path");
    triangle.setAttribute("d", "M1 1 11 7 1 13Z");
    playIcon.append(triangle);
    playButton.append(playIcon, element("span", "", "Play trial"));
    media.append(poster, playButton);

    const details = element("div", "trial-details");
    const heading = element("div", "trial-heading");
    const title = element("h3", "", `Seed ${trial.seed}`);
    title.id = `seed-${trial.seed}`;
    heading.append(title, element("span", "peak-coverage", `Peak ${percentage(trial.peak_coverage)}`));
    const metrics = element("dl", "trial-metrics");
    const finalCoverage = element("div");
    finalCoverage.append(element("dt", "", "Final coverage"), element("dd", "", percentage(trial.final_coverage)));
    metrics.append(finalCoverage);
    const playbackError = element("p", "playback-error");
    playbackError.setAttribute("role", "status");
    playbackError.hidden = true;
    const videoLink = element("a", "", "Open the video directly");
    videoLink.href = trial.video;
    playbackError.append(document.createTextNode("Unable to play this trial. "), videoLink, document.createTextNode("."));
    details.append(heading, metrics, playbackError);
    card.append(media, details);

    // There is no video element or video request before this user action.
    playButton.addEventListener("click", () => {
      const video = element("video", "trial-video");
      video.controls = true;
      video.playsInline = true;
      video.preload = "none";
      video.muted = true;
      video.poster = trial.poster;
      video.setAttribute("aria-label", `Push-T rollout, seed ${trial.seed}, peak coverage ${percentage(trial.peak_coverage)}`);
      video.src = trial.video;
      video.addEventListener("play", () => pauseOtherVideos(video));
      video.addEventListener("error", () => { playbackError.hidden = false; });
      pauseOtherVideos(video);
      media.replaceChildren(video);
      video.focus({ preventScroll: true });
      const playback = video.play();
      if (playback) playback.catch(() => {
        // Native controls remain available if the browser declines playback.
        if (video.error) playbackError.hidden = false;
      });
    }, { once: true });
    return { trial, element: card };
  }

  function applySearch() {
    const query = search.value.trim();
    let visible = 0;
    for (const card of cards) {
      const matches = String(card.trial.seed).includes(query);
      card.element.hidden = !matches;
      if (matches) visible += 1;
      else card.element.querySelector("video")?.pause();
    }
    resultCount.textContent = query ? `${visible} of ${cards.length} trials` : `${cards.length} trials`;
    emptyState.hidden = visible > 0;
  }

  function validTrial(trial) {
    const validCoverage = (value) => Number.isFinite(value) && value >= 0 && value <= 1;
    const safeMediaPath = (value, extension) => typeof value === "string"
      && new RegExp(`^media/pusht/seed_[0-9]+\\.${extension}$`).test(value);
    return Number.isSafeInteger(trial.seed)
      && validCoverage(trial.peak_coverage) && validCoverage(trial.final_coverage)
      && safeMediaPath(trial.video, "mp4") && safeMediaPath(trial.poster, "jpg");
  }

  async function loadTrials() {
    loadError.hidden = true;
    emptyState.hidden = true;
    grid.setAttribute("aria-busy", "true");
    resultCount.textContent = "Loading trials…";
    try {
      const response = await fetch("data/trials.json");
      if (!response.ok) throw new Error(`Trial data returned HTTP ${response.status}`);
      const data = await response.json();
      if (!Array.isArray(data.trials) || !data.trials.length || !data.trials.every(validTrial)
        || new Set(data.trials.map((trial) => trial.seed)).size !== data.trials.length) {
        throw new Error("Invalid trial data");
      }
      const coverages = data.trials.map((trial) => trial.peak_coverage).sort((a, b) => a - b);
      const mean = coverages.reduce((sum, value) => sum + value, 0) / coverages.length;
      const middle = Math.floor(coverages.length / 2);
      const median = coverages.length % 2 ? coverages[middle] : (coverages[middle - 1] + coverages[middle]) / 2;
      document.querySelector("#total-count").textContent = coverages.length;
      document.querySelector("#mean-coverage").textContent = percentage(mean);
      document.querySelector("#median-coverage").textContent = percentage(median);
      cards = [...data.trials].sort((a, b) => a.seed - b.seed).map(createCard);
      grid.replaceChildren(...cards.map((card) => card.element));
      search.disabled = false;
      applySearch();
    } catch (error) {
      resultCount.textContent = "Trials unavailable";
      loadError.hidden = false;
      console.error("Could not load rollout gallery:", error);
    } finally {
      grid.setAttribute("aria-busy", "false");
    }
  }

  search.addEventListener("input", applySearch);
  document.querySelector("#reset-search").addEventListener("click", () => {
    search.value = "";
    applySearch();
    search.focus();
  });
  document.querySelector("#retry-load").addEventListener("click", loadTrials);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) pauseOtherVideos();
  });
  loadTrials();
})();
