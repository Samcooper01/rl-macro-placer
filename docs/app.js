// Shared colour-theme toggle for every page on the site.
(function(){
  var root = document.documentElement;
  try { var saved = localStorage.getItem('rlmp-theme'); if (saved) root.setAttribute('data-theme', saved); } catch(e){}
  var btn = document.getElementById('themeBtn');
  if (!btn) return;
  btn.addEventListener('click', function(){
    var cur = root.getAttribute('data-theme');
    var next = cur === 'dark' ? 'light' : (cur === 'light' ? 'dark' :
      (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'light' : 'dark'));
    root.setAttribute('data-theme', next);
    try { localStorage.setItem('rlmp-theme', next); } catch(e){}
  });
})();

// Figures with a data-dark source swap to it whenever the page is dark.
(function(){
  var root = document.documentElement, mq = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;
  function isDark(){ var t = root.getAttribute('data-theme'); return t ? t === 'dark' : !!(mq && mq.matches); }
  function apply(){
    var dark = isDark();
    document.querySelectorAll('img[data-dark]').forEach(function(img){
      if (!img.dataset.light) img.dataset.light = img.getAttribute('src');
      var want = dark ? img.dataset.dark : img.dataset.light;
      if (img.getAttribute('src') !== want) img.setAttribute('src', want);
    });
  }
  apply();
  new MutationObserver(apply).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
  if (mq && mq.addEventListener) mq.addEventListener('change', apply);
})();
