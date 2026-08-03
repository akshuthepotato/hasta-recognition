document.querySelectorAll('.nav-menu > a').forEach((link) => {
  const submenu = link.parentElement.querySelector('.nav-submenu');
  if (submenu && !submenu.querySelector('[data-lab-link]')) {
    const labLink = link.cloneNode(true);
    labLink.removeAttribute('aria-current');
    labLink.dataset.labLink = 'true';
    submenu.prepend(labLink);
  }
  link.addEventListener('click', (event) => {
    const menu = link.parentElement;
    event.preventDefault();
    const wasOpen = menu.classList.contains('is-open');
    document.querySelectorAll('.nav-menu.is-open').forEach((item) => item.classList.remove('is-open'));
    if (!wasOpen) menu.classList.add('is-open');
  });
});

document.addEventListener('click', (event) => {
  if (!event.target.closest('.nav-menu')) {
    document.querySelectorAll('.nav-menu.is-open').forEach((menu) => menu.classList.remove('is-open'));
  }
});
