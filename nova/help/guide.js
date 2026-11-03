(() => {
  const query = document.getElementById('task-search');
  const tasks = [...document.querySelectorAll('article.task')];
  const navigation = [...document.querySelectorAll('[data-task-link]')];
  const status = document.getElementById('search-status');
  const empty = document.getElementById('search-empty');
  function filter() {
    const terms = query.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    let count = 0;
    tasks.forEach(task => {
      const match = terms.every(term => task.textContent.toLowerCase().includes(term));
      task.hidden = !match;
      navigation.filter(link => link.dataset.taskLink === task.id).forEach(link => link.hidden = !match);
      if (match) count++;
      task.querySelectorAll('details').forEach(detail => {
        if (terms.length && match && terms.every(term => detail.textContent.toLowerCase().includes(term))) {
          if (!detail.open) { detail.open = true; detail.dataset.searchOpened = 'true'; }
        } else if (detail.dataset.searchOpened) { detail.open = false; delete detail.dataset.searchOpened; }
      });
    });
    document.querySelectorAll('.nav-group').forEach(group => {
      group.hidden = !navigation.some(link => link.dataset.group === group.dataset.group && !link.hidden);
    });
    status.textContent = terms.length ? `${count} of ${tasks.length} tasks match. Clear search to show all tasks.` : '';
    empty.hidden = count > 0;
  }
  query.addEventListener('input', filter);
  function reveal(target) {
    if (!target) return;
    const task = target.closest('article.task');
    if (target.hidden || (task && task.hidden)) { query.value = ''; filter(); }
    if (target.tagName === 'DETAILS') target.open = true;
  }
  document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', () => {
    reveal(document.getElementById(link.getAttribute('href').slice(1)));
  }));
  window.addEventListener('hashchange', () => reveal(document.getElementById(location.hash.slice(1))));
  reveal(document.getElementById(location.hash.slice(1)));
  const toast = document.getElementById('copy-status'); let toastTimer;
  function notify(message) {
    toast.textContent = message; toast.hidden = false; window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.hidden = true, 4000);
  }
  document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
    const source = document.getElementById(button.dataset.copy);
    try { await navigator.clipboard.writeText(source.textContent); notify('Example copied. Review it for your experiment.'); }
    catch (_) {
      const range = document.createRange(); range.selectNodeContents(source);
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      notify('Example selected. Use your system Copy command.');
    }
  }));
  const dialog = document.getElementById('image-dialog');
  const image = document.getElementById('full-image');
  const holder = document.getElementById('dialog-image');
  let opener;
  document.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
    opener = button; holder.classList.remove('actual'); image.src = button.dataset.zoom;
    image.alt = button.querySelector('img').alt;
    document.getElementById('image-title').textContent = image.alt;
    document.getElementById('image-caption').textContent = button.closest('figure').querySelector('figcaption').textContent;
    document.getElementById('actual-size').textContent = 'Actual pixels'; dialog.showModal();
  }));
  document.getElementById('close-image').addEventListener('click', () => dialog.close());
  document.getElementById('actual-size').addEventListener('click', event => {
    const actual = holder.classList.toggle('actual'); event.target.textContent = actual ? 'Fit to window' : 'Actual pixels';
  });
  dialog.addEventListener('close', () => { if (opener) opener.focus(); });
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
})();
