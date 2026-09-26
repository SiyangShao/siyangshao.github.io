---
title: Home
---

<div style="display:flex;flex-wrap:wrap;align-items:center;gap:1.5rem;margin-bottom:1.5rem;">
  <img id="site-avatar" src="/images/image.jpeg" alt="Siyang Shao" style="max-width:200px;border-radius:12px;box-shadow:0 2px 8px rgba(0,0,0,0.15);user-select:none;-webkit-user-drag:none;" />
  <div>
    <h1 style="margin:0 0 0.5rem 0;">Siyang Shao</h1>
  </div>
</div>

## Contact
- Email: <siyangshao@gmail.com>
- GitHub: [SiyangShao](https://github.com/SiyangShao)
- LinkedIn: [in/shaosy](https://www.linkedin.com/in/shaosy/)
- Telegram: [@siyangshao](https://t.me/siyangshao)

## Education
- <span id="ms-entry" style="user-select:none;">M.S. in Computer Science, Georgia Institute of Technology, Atlanta, United States, Aug 2025 – Dec 2026</span>
- <span id="beng-entry" style="user-select:none;">B.Eng. in Computer Engineering, Nanyang Technological University, Singapore, Aug 2021 – Jun 2025</span>

## Industry Experience
- Software Engineer Intern, Tiktok, Recommendation Infra, San Jose, United States, May 2026 - Aug 2026
- Software Engineer Intern, Jane Street, Core Infra, Hong Kong SAR, May 2025 – Jul 2025
- Software Engineer Intern, TikTok, Video Infra, Singapore, Jan 2024 – May 2024

## Open Source Contributions
- [ServerlessLLM](https://github.com/ServerlessLLM/ServerlessLLM) | Maintainer & Core Contributor (600+ stars)

## Competitions
- [ICPC](https://icpc.global/ICPCID/B15T259WIX3C) (International Collegiate Programming Contest)
  - Ranked 2nd, Silver Medal, ICPC Asia Manila Regional, 2022
  - Ranked 24th, ICPC Asia Pacific Championship, 2025
  - Ranked 22nd, ICPC Asia Pacific Championship, 2024


<!-- ## Resume 
- [Resume](/resume/main.pdf)
- [Resume (cn)](/resume/main_cn.pdf) -->

<script>
(function () {
  const motto = "所以千錯萬錯都是我的錯; 你是如此傾城又傾國";
  console.log("%c" + motto, "font-size:14px;color:#888;font-family:serif;");

  const avatar = document.getElementById("site-avatar");
  if (!avatar) return;
  avatar.style.cursor = "pointer";

  let clicks = 0;
  let timer;
  avatar.addEventListener("click", function () {
    clicks++;
    clearTimeout(timer);
    timer = setTimeout(function () { clicks = 0; }, 1500);
    if (clicks >= 5) {
      clicks = 0;
      reveal();
    }
  });

  function reveal() {
    const prev = document.getElementById("motto-egg");
    if (prev) prev.remove();
    const el = document.createElement("p");
    el.id = "motto-egg";
    el.textContent = motto;
    el.style.cssText = "opacity:0;transition:opacity 1.2s ease;margin:1rem 0;";
    avatar.parentElement.insertAdjacentElement("afterend", el);
    requestAnimationFrame(function () { el.style.opacity = "1"; });
  }

  const msEntry = document.getElementById("ms-entry");
  if (msEntry) {
    msEntry.addEventListener("click", revealMs);
  }

  function revealMs() {
    const prev = document.getElementById("ms-egg");
    if (prev) prev.remove();
    const el = document.createElement("span");
    el.id = "ms-egg";
    el.innerHTML = "<br> — I decided to drop out of the Ph.D. program. It's too late to have a system Ph.D.";
    el.style.cssText = "opacity:0;transition:opacity 1.2s ease;font-style:italic;color:#888;";
    msEntry.insertAdjacentElement("afterend", el);
    requestAnimationFrame(function () { el.style.opacity = "1"; });
  }

  const bengEntry = document.getElementById("beng-entry");
  if (bengEntry) {
    bengEntry.addEventListener("click", revealBeng);
  }

  function revealBeng() {
    const prev = document.getElementById("beng-egg");
    if (prev) prev.remove();
    const el = document.createElement("span");
    el.id = "beng-egg";
    el.innerHTML = "<br> — After 2 years and 6 months, a paper originally targeting ASPLOS 2025 and focused on ML systems was finally accepted at SoCC 2026. Along the way, both the people involved and the story changed so much that it barely resembles what it once was. The original story and version can now only be found in my undergraduate thesis. Still, I remember it as my first research project—and perhaps the only one I was involved in from start to finish.";
    el.style.cssText = "opacity:0;transition:opacity 1.2s ease;font-style:italic;color:#888;";
    bengEntry.insertAdjacentElement("afterend", el);
    requestAnimationFrame(function () { el.style.opacity = "1"; });
  }
})();
</script>
