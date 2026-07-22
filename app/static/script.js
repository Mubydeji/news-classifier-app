const articleInput = document.getElementById('articleInput');
const charCount = document.getElementById('charCount');
const classifyButton = document.getElementById('classifyButton');
const resultStage = document.getElementById('resultStage');
const resultPending = document.getElementById('resultPending');
const resultComplete = document.getElementById('resultComplete');
const resultError = document.getElementById('resultError');
const resultLabel = document.getElementById('resultLabel');
const probabilityBars = document.getElementById('probabilityBars');
const errorDetail = document.getElementById('errorDetail');

const CATEGORY_COLORS = {
  business: 'var(--cat-business)',
  entertainment: 'var(--cat-entertainment)',
  politics: 'var(--cat-politics)',
  sport: 'var(--cat-sport)',
  tech: 'var(--cat-tech)'
};

const MIN_CHARS = 20;

articleInput.addEventListener('input', () => {
  const len = articleInput.value.length;
  charCount.textContent = `${len} characters`;
  classifyButton.disabled = len < MIN_CHARS;
});

classifyButton.disabled = true;

classifyButton.addEventListener('click', async () => {
  const text = articleInput.value.trim();
  if (text.length < MIN_CHARS) return;

  resultStage.hidden = false;
  resultPending.hidden = false;
  resultComplete.hidden = true;
  resultError.hidden = true;
  classifyButton.disabled = true;

  try {
    const response = await fetch('/classify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.detail || `Request failed (${response.status})`);
    }

    const data = await response.json();
    showResult(data);
  } catch (err) {
    showError(err.message);
  } finally {
    classifyButton.disabled = false;
  }
});

function showResult(data) {
  resultPending.hidden = true;
  resultComplete.hidden = false;

  resultLabel.textContent = data.predicted_category;

  const sortedProbs = Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1]);

  probabilityBars.innerHTML = '';

  for (const [category, prob] of sortedProbs) {
    const isWinner = category === data.predicted_category;
    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-name ${isWinner ? 'is-winner' : ''}">${category}</span>
      <div class="prob-track">
        <div class="prob-fill" style="width: ${(prob * 100).toFixed(1)}%; background: ${CATEGORY_COLORS[category] || 'var(--ink)'};"></div>
      </div>
      <span class="prob-value">${(prob * 100).toFixed(1)}%</span>
    `;
    probabilityBars.appendChild(row);
  }
}

function showError(message) {
  resultPending.hidden = true;
  resultError.hidden = false;
  errorDetail.textContent = message || 'Something went wrong processing this text.';
}
