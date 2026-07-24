document.addEventListener('DOMContentLoaded', () => {
    // 1. Setup Slider value updates in real time
    const sliders = [
        { id: 'ApplicantIncome', suffix: '', prefix: '$', formatComma: true },
        { id: 'CoapplicantIncome', suffix: '', prefix: '$', formatComma: true },
        { id: 'LoanAmount', suffix: 'k', prefix: '$', formatComma: false, multiply: 1000 },
        { id: 'Loan_Amount_Term', suffix: ' Days', prefix: '', formatComma: false }
    ];

    sliders.forEach(sliderInfo => {
        const sliderEl = document.getElementById(sliderInfo.id);
        const valEl = document.getElementById(sliderInfo.id + 'Val');
        
        if (sliderEl && valEl) {
            const updateLabel = () => {
                let value = parseFloat(sliderEl.value);
                
                // Optional multiplication for display details
                if (sliderInfo.multiply) {
                    value = value * 1000;
                }
                
                let displayVal = value;
                if (sliderInfo.formatComma) {
                    displayVal = value.toLocaleString('en-US');
                } else if (sliderInfo.multiply) {
                    displayVal = (value / 1000) + 'k'; // Keep format simple
                }
                
                valEl.textContent = `${sliderInfo.prefix}${displayVal}${sliderInfo.suffix}`;
            };
            
            sliderEl.addEventListener('input', updateLabel);
            updateLabel(); // Initial run
        }
    });

    // 2. Form Submission Handling
    const loanForm = document.getElementById('loanForm');
    const submitBtn = document.getElementById('submitBtn');
    
    // Result panels
    const stateInitial = document.getElementById('stateInitial');
    const stateLoading = document.getElementById('stateLoading');
    const stateResult = document.getElementById('stateResult');
    
    const statusBadge = document.getElementById('statusBadge');
    const statusLabel = document.getElementById('statusLabel');
    const scorePercent = document.getElementById('scorePercent');
    const gaugeCircle = document.getElementById('gaugeCircle');
    const ratioLabel = document.getElementById('ratioLabel');
    const ratioFill = document.getElementById('ratioFill');
    const adviceList = document.getElementById('adviceList');

    loanForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Disable button to prevent double submit
        submitBtn.disabled = true;
        
        // Show Loading panel
        stateInitial.classList.remove('active');
        stateResult.classList.remove('active');
        stateLoading.classList.add('active');
        
        // Collect form data
        const formData = new FormData(loanForm);
        const payload = {};
        formData.forEach((value, key) => {
            payload[key] = value;
        });
        
        try {
            // Send API request
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                throw new Error('Prediction API failed');
            }
            
            const result = await response.json();
            
            if (result.status === 'success') {
                // Populate result details after brief delay for visual effect
                setTimeout(() => {
                    renderResults(result);
                    
                    // Show Result Panel
                    stateLoading.classList.remove('active');
                    stateResult.classList.add('active');
                    submitBtn.disabled = false;
                }, 800);
            } else {
                alert('Inference error: ' + result.message);
                resetToInitial();
            }
            
        } catch (error) {
            console.error('Request failed:', error);
            alert('Underwriting Server is offline. Please check app.py terminal.');
            resetToInitial();
        }
    });

    function resetToInitial() {
        stateLoading.classList.remove('active');
        stateResult.classList.remove('active');
        stateInitial.classList.add('active');
        submitBtn.disabled = false;
    }

    function renderResults(data) {
        const isApproved = data.prediction === 1;
        
        // 1. Set Status Badge
        if (isApproved) {
            statusBadge.className = 'status-badge status-approved';
            statusLabel.textContent = 'APPROVED';
        } else {
            statusBadge.className = 'status-badge status-rejected';
            statusLabel.textContent = 'REJECTED';
        }
        
        // 2. Animate Circular Radial Gauge
        const probability = data.probability;
        const targetPercent = Math.round(probability * 100);
        
        // Stroke dasharray of our SVG circle is 440 (perimeter = 2 * PI * 70 = ~439.8)
        const circumference = 440;
        const offset = circumference - (probability * circumference);
        gaugeCircle.style.strokeDashoffset = offset;
        
        // Adjust gauge color depending on approval status
        if (isApproved) {
            gaugeCircle.style.stroke = '#10b981'; // Emerald Green
        } else {
            gaugeCircle.style.stroke = '#f43f5e'; // Rose Red
        }
        
        // Count up animation for percent number
        let currentPercent = 0;
        const duration = 1000; // 1s
        const stepTime = Math.abs(Math.floor(duration / targetPercent));
        scorePercent.textContent = '0%';
        
        const timer = setInterval(() => {
            currentPercent++;
            scorePercent.textContent = currentPercent + '%';
            if (currentPercent >= targetPercent) {
                clearInterval(timer);
                scorePercent.textContent = targetPercent + '%';
            }
        }, stepTime);

        // 3. Set Loan-to-Income progress bar
        const ratioPercent = (data.loan_to_income_ratio * 100).toFixed(1);
        ratioLabel.textContent = ratioPercent + '%';
        
        // Cap visual fill at 100%
        const fillWidth = Math.min(100, data.loan_to_income_ratio * 100);
        ratioFill.style.width = fillWidth + '%';
        
        if (data.loan_to_income_ratio > 0.4) {
            ratioFill.style.background = 'var(--error)';
        } else {
            ratioFill.style.background = 'linear-gradient(90deg, var(--primary), var(--secondary))';
        }

        // 4. Render Advice list items
        adviceList.innerHTML = '';
        data.advice.forEach(item => {
            const li = document.createElement('li');
            li.textContent = item;
            
            // Apply contextual styling classes
            if (item.startsWith('CRITICAL') || item.startsWith('HIGH RISK')) {
                li.className = 'critical-advice';
            } else if (item.startsWith('EXCELLENT') || item.startsWith('STABLE')) {
                li.className = 'success-advice';
            }
            
            adviceList.appendChild(li);
        });
    }
});
