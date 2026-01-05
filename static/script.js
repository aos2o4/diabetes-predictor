document.getElementById('prediction-form').addEventListener('submit', async function (e) {
    e.preventDefault();

    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());
    const btn = document.getElementById('predict-btn');
    const originalBtnText = btn.innerText;

    btn.innerText = 'جاري المعالجة...';
    btn.disabled = true;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();

        if (response.ok) {
            const resultSection = document.getElementById('result-section');
            const resultText = document.getElementById('result-text');
            const probabilityText = document.getElementById('probability-text');

            resultSection.classList.remove('hidden');

            if (result.prediction.includes("Positive")) {
                resultText.innerHTML = `<span class="result-positive">مصاب بالسكري (Positive)</span>`;
            } else {
                resultText.innerHTML = `<span class="result-negative">غير مصاب (Negative)</span>`;
            }

            probabilityText.innerText = result.probability;
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('حدث خطأ أثناء الاتصال بالخادم');
    } finally {
        btn.innerText = originalBtnText;
        btn.disabled = false;
    }
});
