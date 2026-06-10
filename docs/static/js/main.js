/* DisPOSE project page — interactivity
   - theme toggle with localStorage + system preference
   - IntersectionObserver scroll reveal
   - BibTeX copy-to-clipboard
   - GDPR banner show/dismiss
*/

(() => {
  "use strict";

  /* ---------- Theme toggle ---------- */
  const STORAGE_KEY = "dispose-theme";
  const root = document.documentElement;

  const getSystemTheme = () =>
    window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";

  const applyTheme = (theme) => {
    root.setAttribute("data-theme", theme);
    const btn = document.getElementById("theme-toggle");
    if (btn) btn.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
  };

  const stored = localStorage.getItem(STORAGE_KEY);
  applyTheme(stored || getSystemTheme());

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("theme-toggle");
    if (btn) {
      btn.addEventListener("click", () => {
        const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
        applyTheme(next);
        localStorage.setItem(STORAGE_KEY, next);
      });
    }

    // Follow system preference only if user hasn't set one explicitly.
    if (!stored) {
      window
        .matchMedia("(prefers-color-scheme: dark)")
        .addEventListener("change", (e) => applyTheme(e.matches ? "dark" : "light"));
    }
  });

  /* ---------- Scroll reveal ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const els = document.querySelectorAll(".reveal");
    if (reduce || !("IntersectionObserver" in window)) {
      els.forEach((el) => el.classList.add("visible"));
      return;
    }
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("visible");
            io.unobserve(e.target);
          }
        });
      },
      { rootMargin: "0px 0px -60px 0px", threshold: 0.05 }
    );
    els.forEach((el) => io.observe(el));
  });

  /* ---------- Scroll-spy: highlight the nav link of the section in view ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    const links = Array.from(document.querySelectorAll(".navbar-links a[href^='#']"));
    if (!links.length || !("IntersectionObserver" in window)) return;
    // Map each nav target id to its link, and collect the observed elements.
    const linkFor = new Map();
    const targets = [];
    links.forEach((a) => {
      const id = a.getAttribute("href").slice(1);
      const el = document.getElementById(id);
      if (el) {
        linkFor.set(id, a);
        targets.push(el);
      }
    });
    if (!targets.length) return;

    const visible = new Set();
    const setActive = () => {
      // Choose the topmost currently-visible section.
      let best = null;
      let bestTop = Infinity;
      visible.forEach((id) => {
        const top = document.getElementById(id).getBoundingClientRect().top;
        if (top < bestTop) { bestTop = top; best = id; }
      });
      links.forEach((a) => a.classList.remove("active"));
      if (best && linkFor.has(best)) linkFor.get(best).classList.add("active");
    };

    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          const id = e.target.id;
          if (e.isIntersecting) visible.add(id);
          else visible.delete(id);
        });
        setActive();
      },
      // Activate a section once it crosses the upper third, below the sticky bar.
      { rootMargin: "-45% 0px -50% 0px", threshold: 0 }
    );
    targets.forEach((el) => io.observe(el));
  });

  /* ---------- BibTeX copy ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("copy-bib");
    if (!btn) return;
    btn.addEventListener("click", async () => {
      const pre = document.querySelector(".bibtex-container pre code");
      if (!pre) return;
      try {
        await navigator.clipboard.writeText(pre.innerText);
        const label = btn.querySelector(".copy-label");
        const orig = label.textContent;
        label.textContent = "Copied!";
        btn.classList.add("copied");
        setTimeout(() => {
          label.textContent = orig;
          btn.classList.remove("copied");
        }, 1600);
      } catch (err) {
        // Fallback: select the text so the user can ctrl+C manually
        const range = document.createRange();
        range.selectNodeContents(pre);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
    });
  });

  /* ---------- Tab systems (dataset switcher, generalization switcher, …) ---------- */
  // Each tab group is wrapped in a .tab-system; tabs and panels are scoped to
  // that wrapper, so multiple independent tab systems coexist on one page.
  document.addEventListener("DOMContentLoaded", () => {
    const groups = document.querySelectorAll(".tab-system");
    const panelIdToGroup = new Map();
    if (!groups.length) return;

    groups.forEach((group) => {
      const tabs = Array.from(group.querySelectorAll(".dataset-tab-btn"));
      const panels = group.querySelectorAll(".dataset-panel");
      if (!tabs.length || !panels.length) return;

      const activate = (target, { scroll = false, focus = false } = {}) => {
        tabs.forEach((t) => {
          const on = t.dataset.target === target;
          t.classList.toggle("active", on);
          t.setAttribute("aria-selected", on ? "true" : "false");
          // Roving tabindex: only the active tab is in the tab order.
          t.tabIndex = on ? 0 : -1;
          if (on && focus) t.focus();
        });
        panels.forEach((p) => p.classList.toggle("active", p.id === target));
        if (scroll) {
          const bar = group.querySelector(".dataset-tabs");
          if (bar) bar.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      };

      tabs.forEach((t, i) => {
        t.addEventListener("click", () => activate(t.dataset.target));
        // Arrow / Home / End keyboard navigation, per the ARIA tablist pattern.
        t.addEventListener("keydown", (e) => {
          let j = null;
          if (e.key === "ArrowRight" || e.key === "ArrowDown") j = (i + 1) % tabs.length;
          else if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = (i - 1 + tabs.length) % tabs.length;
          else if (e.key === "Home") j = 0;
          else if (e.key === "End") j = tabs.length - 1;
          if (j === null) return;
          e.preventDefault();
          activate(tabs[j].dataset.target, { focus: true });
        });
      });
      panels.forEach((p) => panelIdToGroup.set(p.id, { group, activate }));

      // Initialize active panel: respect a URL hash that targets this group, else first panel.
      const hash = window.location.hash.slice(1);
      const targets = Array.from(panels).map((p) => p.id);
      activate(targets.includes(hash) ? hash : targets[0]);
    });

    // Global hashchange — if the new hash is a panel id we know about, switch
    // its tab group to it and scroll into view.
    window.addEventListener("hashchange", () => {
      const h = window.location.hash.slice(1);
      const entry = panelIdToGroup.get(h);
      if (entry) entry.activate(h, { scroll: true });
    });
  });

  /* ---------- Lazy-load <video data-src=...> ----------
     The MM-OR showcase video lives well below the fold. Defer its fetch until
     the element is about to enter the viewport so we don't burn bandwidth on
     scrollers who never reach it.
  */
  document.addEventListener("DOMContentLoaded", () => {
    const videos = document.querySelectorAll("video[data-src]");
    if (!videos.length) return;

    const hydrate = (v) => {
      if (!v.dataset.src) return;
      v.src = v.dataset.src;
      v.load();
      // Autoplay/muted attrs trigger playback when ready; ignore the rejection
      // if the browser blocks it (will resume on first user interaction).
      const p = v.play();
      if (p && typeof p.catch === "function") p.catch(() => {});
      delete v.dataset.src;
    };

    if (!("IntersectionObserver" in window)) {
      videos.forEach(hydrate);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (e.isIntersecting) {
          hydrate(e.target);
          io.unobserve(e.target);
        }
      }),
      { rootMargin: "300px 0px" }
    );
    videos.forEach((v) => io.observe(v));
  });

  /* ---------- Animated counters ----------
     Any element with [data-count="<final>"] will animate from 0 to <final>
     when it first enters the viewport. Optional attrs:
       data-decimals="1"      -> render with this many decimal places
       data-duration="900"    -> tween duration in ms
       data-suffix=" mAP"     -> appended after the number on every tick
  */
  document.addEventListener("DOMContentLoaded", () => {
    const targets = document.querySelectorAll("[data-count]");
    if (!targets.length) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const animate = (el) => {
      const final = parseFloat(el.dataset.count);
      if (isNaN(final)) return;
      const decimals = parseInt(el.dataset.decimals || "0", 10);
      const duration = parseInt(el.dataset.duration || "900", 10);
      const suffix = el.dataset.suffix || "";

      if (reduce) {
        el.textContent = final.toFixed(decimals) + suffix;
        return;
      }
      const start = performance.now();
      const ease = (t) => 1 - Math.pow(1 - t, 3); // ease-out cubic
      const tick = (now) => {
        const t = Math.min(1, (now - start) / duration);
        const v = final * ease(t);
        el.textContent = v.toFixed(decimals) + suffix;
        if (t < 1) requestAnimationFrame(tick);
      };
      requestAnimationFrame(tick);
    };

    if (!("IntersectionObserver" in window)) {
      targets.forEach(animate);
      return;
    }
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (e.isIntersecting) {
          animate(e.target);
          io.unobserve(e.target);
        }
      }),
      { threshold: 0.4 }
    );
    targets.forEach((el) => io.observe(el));
  });

  /* ---------- Teaser click-to-pause ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".teaser-video-wrap").forEach((wrap) => {
      const video = wrap.querySelector("video");
      if (!video) return;
      const setPaused = () => wrap.classList.toggle("paused", video.paused || video.ended);
      const atEnd = () => video.ended || (video.duration > 0 && video.currentTime >= video.duration - 0.05);
      const toggle = () => {
        if (video.paused || video.ended) {
          if (atEnd()) video.currentTime = 0;          // finished -> replay from the start
          const p = video.play();
          if (p && p.catch) p.catch(() => {});         // ignore autoplay-policy rejections
        } else {
          video.pause();
        }
      };
      wrap.addEventListener("click", toggle);
      // role="button" must respond to Enter/Space for keyboard users.
      wrap.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
          e.preventDefault();
          toggle();
        }
      });
      video.addEventListener("play", setPaused);
      video.addEventListener("pause", setPaused);
      video.addEventListener("ended", setPaused);   // pause on the last frame, show the replay glyph
      setPaused();
    });
  });

  /* ---------- Lightbox ---------- */
  document.addEventListener("DOMContentLoaded", () => {
    const lightbox = document.getElementById("lightbox");
    if (!lightbox) return;
    const lbImg = lightbox.querySelector("img");
    const lbCaption = lightbox.querySelector(".lightbox-caption");
    const lbClose = lightbox.querySelector(".lightbox-close");

    // Remember what opened the lightbox so focus can return there on close.
    let lastTrigger = null;

    const open = (src, alt, captionHtml, trigger) => {
      lastTrigger = trigger || null;
      lbImg.src = src;
      lbImg.alt = alt || "";
      lbCaption.innerHTML = captionHtml || alt || "";
      lightbox.classList.add("visible");
      lightbox.setAttribute("aria-hidden", "false");
      document.body.style.overflow = "hidden";
      lbClose.focus();
    };
    const close = () => {
      lightbox.classList.remove("visible");
      lightbox.setAttribute("aria-hidden", "true");
      document.body.style.overflow = "";
      if (lastTrigger && typeof lastTrigger.focus === "function") lastTrigger.focus();
      lastTrigger = null;
    };

    // Make an image open the lightbox by click OR keyboard (Enter/Space), and
    // expose it to assistive tech as a button.
    const makeZoomable = (img, getCaption) => {
      img.classList.add("zoomable");
      img.style.cursor = "zoom-in";
      img.setAttribute("role", "button");
      img.setAttribute("tabindex", "0");
      if (img.alt) img.setAttribute("aria-label", `Enlarge image: ${img.alt}`);
      const trigger = () => open(img.src, img.alt, getCaption(), img);
      img.addEventListener("click", trigger);
      img.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          trigger();
        }
      });
    };

    document.querySelectorAll(".figure img").forEach((img) => {
      // Pull caption from the figure's .fig-caption if present, otherwise from alt.
      makeZoomable(img, () => {
        const fig = img.closest("figure");
        const cap = fig ? fig.querySelector(".fig-caption") : null;
        return cap ? cap.innerHTML : "";
      });
    });

    // Teaser uses .teaser instead of .figure
    document.querySelectorAll(".teaser img").forEach((img) => {
      makeZoomable(img, () => {
        const cap = img.closest(".teaser").querySelector(".teaser-caption");
        return cap ? cap.innerHTML : "";
      });
    });

    lbClose.addEventListener("click", close);
    lightbox.addEventListener("click", (e) => {
      if (e.target === lightbox || e.target === lbClose) close();
    });
    document.addEventListener("keydown", (e) => {
      if (!lightbox.classList.contains("visible")) return;
      if (e.key === "Escape") {
        close();
      } else if (e.key === "Tab") {
        // Only the close button is focusable inside the overlay — keep focus on it.
        e.preventDefault();
        lbClose.focus();
      }
    });
  });

  /* ---------- GDPR banner ---------- */
  const GDPR_KEY = "dispose-gdpr-ack";
  document.addEventListener("DOMContentLoaded", () => {
    const banner = document.getElementById("gdpr-banner");
    if (!banner) return;
    if (localStorage.getItem(GDPR_KEY)) return;
    // Reveal after a brief delay so it doesn't pop in before the hero settles.
    setTimeout(() => banner.classList.add("visible"), 900);
    const ack = document.getElementById("accept-gdpr");
    if (ack) {
      ack.addEventListener("click", () => {
        banner.classList.remove("visible");
        localStorage.setItem(GDPR_KEY, "1");
      });
    }
  });
})();
