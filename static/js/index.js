// OccDirector project page — minimal interactions.
// Based on the Nerfies project page template, trimmed to what this page uses.

document.addEventListener('DOMContentLoaded', function () {
  // Mobile navbar burger toggle (Bulma).
  document.querySelectorAll('.navbar-burger').forEach(function (burger) {
    burger.addEventListener('click', function () {
      var target = document.getElementById(burger.dataset.target);
      burger.classList.toggle('is-active');
      if (target) {
        target.classList.toggle('is-active');
      }
    });
  });
});
