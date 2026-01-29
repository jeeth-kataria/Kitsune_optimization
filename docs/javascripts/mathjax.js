// MathJax configuration
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    ignoreHtmlClass: '.*|',
    processHtmlClass: 'arithmatex'
  }
};

// Load MathJax
document$.subscribe(() => {
  MathJax.typesetPromise()
});
