/*
 * Page behaviour shared by every page (replaces the parts of Bootstrap's JavaScript the app used):
 * - the menu slides in over the page on small screens (Escape, the close button or
 *   the dark backdrop close it, and focus returns to the menu button);
 * - message boxes with a close button can be dismissed.
 */
(function () {
  "use strict";
  const sidebar = document.getElementById("sidebar");
  const openButton = document.getElementById("sidebar-open");
  const closeButton = document.getElementById("sidebar-close");
  const backdrop = document.getElementById("sidebar-backdrop");

  function setMenu(open) {
    if (!sidebar) return;
    sidebar.classList.toggle("-translate-x-full", !open);
    sidebar.classList.toggle("invisible", !open); // hidden from keyboard and screen readers while closed
    backdrop.classList.toggle("hidden", !open);
    openButton.setAttribute("aria-expanded", String(open));
    (open ? closeButton : openButton).focus();
  }

  if (sidebar && openButton) {
    openButton.addEventListener("click", () => setMenu(true));
    closeButton.addEventListener("click", () => setMenu(false));
    backdrop.addEventListener("click", () => setMenu(false));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !sidebar.classList.contains("invisible") && openButton.offsetParent !== null) setMenu(false);
    });
  }

  document.addEventListener("click", (e) => {
    const button = e.target.closest("[data-dismiss]");
    if (button) button.closest(button.dataset.dismiss)?.remove();
  });
})();
