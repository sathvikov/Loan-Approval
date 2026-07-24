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

    // 2. Trained Logistic Regression Model Parameters (Extracted from Python Model)
    const MODEL_PARAMS = {
        intercept: 0.8085659513910706,
        coefficients: [
            -0.10641613549990724, // Gender (Female=0, Male=1)
            0.24818273591054524,  // Married (No=0, Yes=1)
            -0.17028397431189526, // Education (Graduate=0, Not Graduate=1)
            -0.06712319726840615, // Self_Employed (No=0, Yes=1)
            0.051680631776285145, // Property_Area (Rural=0, Semiurban=1, Urban=2)
            1.2345759986075555,   // Credit_History (0 or 1)
            0.07736556727423251,  // Dependents_Numeric
            -0.31284563024666767, // TotalIncome_Log
            0.28784016493549613,  // LoanAmount_Log
            0.02342529940089191,  // Loan_Amount_Term
            -0.29390510543421206, // Loan_Income_Ratio
            -0.08951311733905182   // Income_Per_Dependent
        ],
        mean: [
            0.8228105906313645, 0.6558044806517311, 0.22606924643584522, 0.13441955193482688,
            1.034623217922607, 0.8635437881873728, 0.7494908350305499, 8.67802731590134,
            4.86954654887553, 341.74338085539716, 0.023587390971882168, 5005.5400882469785
        ],
        scale: [
            0.38182891794130724, 0.4751052134094556, 0.41828452308419717, 0.341102530029414,
            0.7899720344831256, 0.3432723613551995, 1.0012084070541567, 0.542800087310027,
            0.4944640917855741, 65.04160764484553, 0.008082600190249797, 4579.57494136166
        ]
    };

    // 3. Form Submission Handling (Run Client-Side ML Inference)
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

    loanForm.addEventListener('submit', (e) => {
        e.preventDefault();
        
        // Disable button to prevent double submit
        submitBtn.disabled = true;
        
        // Show Loading panel
        stateInitial.classList.remove('active');
        stateResult.classList.remove('active');
        stateLoading.classList.add('active');
        
        // Gather input values
        const inputs = {
            Gender: document.getElementById('Gender').value,
            Married: document.getElementById('Married').value,
            Dependents: document.getElementById('Dependents').value,
            Education: document.getElementById('Education').value,
            Self_Employed: document.getElementById('Self_Employed').value,
            Property_Area: document.getElementById('Property_Area').value,
            Credit_History: parseFloat(document.getElementById('Credit_History').value),
            ApplicantIncome: parseFloat(document.getElementById('ApplicantIncome').value),
            CoapplicantIncome: parseFloat(document.getElementById('CoapplicantIncome').value),
            LoanAmount: parseFloat(document.getElementById('LoanAmount').value),
            Loan_Amount_Term: parseFloat(document.getElementById('Loan_Amount_Term').value)
        };
        
        // Perform local prediction
        try {
            const results = runInference(inputs);
            
            // Populate result details after brief delay for visual effect
            setTimeout(() => {
                renderResults(results);
                
                // Show Result Panel
                stateLoading.classList.remove('active');
                stateResult.classList.add('active');
                submitBtn.disabled = false;
            }, 600);
            
        } catch (error) {
            console.error('Inference failed:', error);
            alert('An error occurred during inference calculation.');
            resetToInitial();
        }
    });

    function resetToInitial() {
        stateLoading.classList.remove('active');
        stateResult.classList.remove('active');
        stateInitial.classList.add('active');
        submitBtn.disabled = false;
    }

    // Client-side Machine Learning Inference Engine
    function runInference(inputs) {
        // A. Encode Categorical inputs
        const gender_enc = inputs.Gender === 'Male' ? 1 : 0;
        const married_enc = inputs.Married === 'Yes' ? 1 : 0;
        const edu_enc = inputs.Education === 'Graduate' ? 0 : 1;
        const self_enc = inputs.Self_Employed === 'Yes' ? 1 : 0;
        
        let prop_enc = 1; // Default Semiurban
        if (inputs.Property_Area === 'Rural') prop_enc = 0;
        if (inputs.Property_Area === 'Urban') prop_enc = 2;

        const dependents_num = inputs.Dependents === '3+' ? 3 : parseInt(inputs.Dependents);
        const credit_hist = inputs.Credit_History;

        // B. Feature Engineering
        const total_income = inputs.ApplicantIncome + inputs.CoapplicantIncome;
        const loan_income_ratio = inputs.LoanAmount / (total_income || 1.0);
        const income_per_dep = total_income / (dependents_num + 1);

        const app_income_log = Math.log1p(inputs.ApplicantIncome);
        const total_income_log = Math.log1p(total_income);
        const loan_amount_log = Math.log1p(inputs.LoanAmount);

        // C. Construct Feature Vector in the exact order as Python model
        const X = [
            gender_enc,
            married_enc,
            edu_enc,
            self_enc,
            prop_enc,
            credit_hist,
            dependents_num,
            total_income_log,
            loan_amount_log,
            inputs.Loan_Amount_Term,
            loan_income_ratio,
            income_per_dep
        ];

        // D. Apply StandardScaler scaling parameters: (X - Mean) / Scale
        const X_scaled = [];
        for (let i = 0; i < X.length; i++) {
            X_scaled.push((X[i] - MODEL_PARAMS.mean[i]) / MODEL_PARAMS.scale[i]);
        }

        // E. Compute Logistic Regression linear combination: z = Intercept + sum(coef * X_scaled)
        let z = MODEL_PARAMS.intercept;
        for (let i = 0; i < X_scaled.length; i++) {
            z += X_scaled[i] * MODEL_PARAMS.coefficients[i];
        }

        // F. Compute Sigmoid function: prob = 1 / (1 + e^-z)
        const probability = 1 / (1 + Math.exp(-z));
        const prediction = probability >= 0.5 ? 1 : 0;

        // G. Generate advice list
        const advice = [];
        if (credit_hist === 0) {
            advice.append ? advice.push("CRITICAL: Lack of credit history or poor repayment record is the primary risk factor. Rebuilding your credit score will yield the highest chance of approval.") : advice.push("CRITICAL: Lack of credit history or poor repayment record is the primary risk factor. Rebuilding your credit score will yield the highest chance of approval.");
        }
        if (loan_income_ratio > 0.4) {
            advice.push(`HIGH RISK: The requested loan amount ($${inputs.LoanAmount}k) is high relative to your monthly household income ($${total_income}). A loan-to-income ratio of ${(loan_income_ratio * 100).toFixed(1)}% indicates high debt burden. Try applying for a lower loan amount.`);
        }
        if (inputs.Self_Employed === 'Yes' && inputs.ApplicantIncome < 3000) {
            advice.push("RISK FACTOR: Self-employed applicants with lower primary incomes are classified as higher risk. Providing secondary collateral or co-signers is highly recommended.");
        }

        if (prediction === 1) {
            if (advice.length === 0) {
                advice.push("EXCELLENT PROFILE: Your financial profile, stable income, and positive credit history align perfectly with approval criteria.");
            } else {
                advice.push("STABLE PROFILE: Despite minor risk factors, your overall profile remains strong enough for standard approval.");
            }
        } else {
            if (advice.length === 0) {
                advice.push("INSUFFICIENT MARGIN: Your overall income and requested loan size fall just short of safety thresholds. Consider reducing the loan amount.");
            }
        }

        return {
            prediction: prediction,
            probability: probability,
            loan_to_income_ratio: loan_income_ratio,
            advice: advice
        };
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
        
        // Stroke dasharray of our SVG circle is 440
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
        const duration = 800; // 0.8s
        const stepTime = Math.max(5, Math.floor(duration / (targetPercent || 1)));
        scorePercent.textContent = '0%';
        
        const timer = setInterval(() => {
            if (targetPercent === 0) {
                scorePercent.textContent = '0%';
                clearInterval(timer);
                return;
            }
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
